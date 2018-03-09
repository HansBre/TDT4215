def rate_article(user_keywords, article):
    rating = 1
    if int(article['time']) < 20:
        # If user quickly clicked away from article, we assume user did not like article at all, and give it the
        # lowest possible rating.
        return rating
    else:
        # User has spent time on the article.
        rating = 2
        for k in article['keywords']:
            if k in user_keywords:
                if user_keywords[k] > rating:
                    rating = user_keywords[k]
                    if rating == 5:
                        return rating
    return rating
