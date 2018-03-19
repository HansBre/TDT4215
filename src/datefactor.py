from datetime import datetime
from surprise import AlgoBase, PredictionImpossible


class DateFactor(AlgoBase):
    def __init__(self, db, threshold_date=None, cut_after=None, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        # Note: We are not saving db. This way, this object may be pickled.
        self.dates = {row['id']: row['published'] for row in db.get_all().values() if row['published']}
        self.oldest_date = min(self.dates.values())
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
