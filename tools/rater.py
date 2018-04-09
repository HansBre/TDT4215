def rate_article(user_keywords, article):
    rating = 1
    time_spent = int(article['time'])
    if time_spent < 20:
        # If user quickly clicked away from article, we assume user did not like article at all, and give it the
        # lowest possible rating.
        return rating
    if time_spent > 20:
        rating = 2
    if time_spent > 40:
        rating = 3
    if time_spent > 60:
        rating = 4
    if time_spent > 80:
        rating = 5
    return rating

