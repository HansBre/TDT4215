from datetime import datetime
from surprise import AlgoBase, PredictionImpossible


class DateFactor(AlgoBase):
    def __init__(
            self,
            db,
            threshold_date=None,
            cut_after=None,
            weight=1.0,
            oldest_date=None,
            **kwargs
    ):
        """
        Algorithm which weights articles based on their publication date.

        ^ upper                                              _______
        R                                                   /¦     |
        a                                                  / ¦     |
        t                                                 /  ¦     |
        i                                                /   ¦     |
        n                                               /    ¦     |
        g weight ______________________________________/     ¦     |
          lower                                       ¦      ¦     |________
        Time --->                                     ¦      ¦     ¦
                                                oldest_date  ¦ cut_after
                                                      threshold_date

        Args:
            db: Instance of FulltextDb. Used to find article's publication date.
            threshold_date: datetime which marks the first date which is not
                weighted down because of publication date. Defaults to January
                1st 2017, since that's the first day of the dataset.
            cut_after: datetime which marks the first date to weight as low as
                possible. By default, no such weighting down is done. This is
                useful to eliminate articles which are published after the
                data set.
            weight: Decides how much of the rating scale is used. Defaults to
                the entire scale, but set this to a float between 0 and 1 to
                adjust the scale upwards. For example, 0.5 makes it so only the
                upper half of the rating scale is used. This is only used for
                dates before the threshold_date.
            oldest_date: The oldest date to weight as minimum. Defaults to the
                first known publication date. Use this to adjust the rate by
                which articles are weighted down.
            **kwargs: Extra arguments passed to the AlgoBase constructor.
        """
        super().__init__(**kwargs)
        # Note: We are not saving db. This way, this object may be pickled.
        self.dates = {row['id']: row['published'] for row in db.get_all().values() if row['published']}
        self.oldest_date = oldest_date or min(self.dates.values())
        self.threshold_date = threshold_date or datetime(2017, 1, 1)
        self.date_scale = self.threshold_date - self.oldest_date
        self.trainset = None
        self.upper = None
        self.lower = None
        self.range = None
        self.cut_after = cut_after
        self.weight = weight

    def fit(self, trainset):
        self.trainset = trainset
        self.lower, self.upper = trainset.rating_scale
        self.range = self.upper - self.lower

    def estimate(self, u, i):
        if self.trainset.knows_item(i):
            item_id = self.trainset.to_raw_iid(i)
        else:
            # Strip off "UNK__" prefix to obtain the raw iid
            item_id = i[5:]

        try:
            published_date = self.dates[item_id]
        except KeyError:
            raise PredictionImpossible('No publication date registered')

        # Is there an upper bound? We use this to avoid recommending
        # "impossible" articles
        if self.cut_after:
            if self.cut_after < published_date:
                return self.lower

        if self.threshold_date < published_date:
            return self.upper

        diff = published_date - self.oldest_date
        # rating should be in domain [0, 1]
        rating = diff / self.date_scale

        # We may not want to straight up exclude the oldest stuff
        weighted_rating = (1.0 - self.weight) + (rating * self.weight)
        # Convert to scale used by the rest of the algorithms
        return self.lower + (self.range * weighted_rating)
