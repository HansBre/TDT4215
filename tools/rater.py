def rate_article(user, article):
    if int(article['time']) < 20:
        return 1
    else:
        return 5
