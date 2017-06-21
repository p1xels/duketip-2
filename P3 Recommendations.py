# -*- coding: utf-8 -*-
import pandas as pnd
import numpy as np

########################################################
# These functions will be used in Phase 3 (skip for now)
########################################################

def findSimilarity(iLike, userLikes):
    # Create an And similarity
    similarityAnd = np.logical_and(iLike, userLikes) # replace 0 with the correct code
    # Create an Or similarity
    similarityOr = np.logical_or(iLike, userLikes) # replace 0 with the correct code
    # Calculate the similarity
    userSimilarity = similarityAnd.sum() / float(similarityOr.sum())

    if np.array_equal(iLike,userLikes): return 0 # no exact matches
    # Return the index of the user which is the best match
    return userSimilarity

def printMovie(id):
    # Print the id of the movie and the name.  This should look something like
    # "    - 430: Duck Soup (1933)" if the id is 430 and the name is Duck Soup (1933)
    print("\t- %d: %s" % (id, movieNames[id]))

def processLikes(iLike):
    iLikeNp = np.zeros(maxMovie)

    print "\n\nSince you like:"

    # Print the name of each movie the user reported liking
    # Hint: Use a for loop and the printMovie function.
    for movie in iLike:
        iLikeNp[movie] = 1
        printMovie(movie)

    # Find the most similar user
    similarities = np.apply_along_axis(lambda x:findSimilarity(iLikeNp,x), 1, userLikes)
    maxsimilar = np.argmax(similarities)
    user = {'id': maxsimilar, 'sim': similarities[maxsimilar], 'likes': userLikes[maxsimilar]}
    print "Similarity found: %.8f (User ID: %d)" % (user['sim'], user['id'])
    print "\nYou might like: "
    # Find the indexes of the values that are ones
    # https://stackoverflow.com/a/17568803/3854385 (Note: You don't want it to be a list, but you do want to flatten it.)
    recLikes = np.argwhere(user['likes'] == np.amax(user['likes'])).flatten()

    # For each item the similar user likes that the person didn't already say they liked
    # print the movie name using printMovie (you'll also need a for loop and an if statement)
    for movie in recLikes:
        if iLikeNp[movie]: continue
        printMovie(movie)

def starRating(rate):
    # Returns the star rating for a integer below 5
    rating = int(rate)
    if rating > 5: raise ValueError("Attempted to rate something %d out of 5" % rating)
    return '★' * rating + \
           '☆' * (5 - rating)

def printMovieListing(index, movie):
    """"
    Outputs a rating
    Example:
    16. Close Shave, A (1995) (ID: 408)
        Rating: 4.49 [★★★★☆] (112 reviews)
    """
    movID = movie['movie']
    indent = ' ' * (len(str(i)) + 2) # calculates '[#]. ' length
    print "%d. %s (ID: %d)\n%sRating: %.2f [%s] (%d reviews)" % (i, movieNames[movID], movID,
                                                                 indent, movie['rating'], starRating(movie['rating']), movie['count'])
########################################################
# Begin Phase 1
########################################################

# Load Data
# Load the movie names data (u.item) with just columns 0 and 1 (id and name) id is np.int, name is S128
# Create a dictionary with the ids as keys and the names as the values
movieNames = pnd.read_csv('./ml-100k/u.item', delimiter='|', usecols=[0, 1], header=None, index_col=0).to_dict()[1] # replace 0 with the code to make the dict

# Load the movie Data (u.data) with just columns 0, 1, and 2 (user, movie, rating) all are np.int
movieData = pnd.read_csv('./ml-100k/u.data', delimiter='\t', usecols=[0,1,2], names=['user','movie','rating'], header=None) # replace 0 with the correct cod eto load the movie data

print movieData
print movieNames
#exit(0) # Delete this after we finish phase 1, for now just get the data loaded

########################################################
# Begin Phase 2
########################################################

# Compute average rating per movie

# This looks horrible, but trust me, it just works
movieRating = movieData.groupby('movie', as_index=False)['rating'].mean().sort_values('rating', ascending=False).reset_index(drop=True)
movieRating.index = movieRating.index + 1
# Basically, group movie data by movies, average the ratings, and then sort the data by rating.
# As well, we reset the index, and make it one-indexed, but that's not important.

movieRatingCount = movieData.groupby('movie').size().to_dict()
# Basically group by movies and then create a dictionary listing the size of each group

# Merge movieRatingCount into movieRating because it makes sense
movieRating['count'] = movieRating['movie'].map(movieRatingCount)

# Top 10 Movies
print "Top Ten Movies:"
# Print the top 10 movies
# It should print the number, title, id, rating and count of reviews for each movie
# ie 2. Someone Else's America (1995) (ID: 1599) Rating: 5.0 Count: 1
for i,movie in movieRating.head(10).iterrows():
    printMovieListing(i,movie)

# Top 10 Movies with at least 100 ratings
print("\n\nTop Ten movies with at least 100 ratings:")
for i,movie in movieRating.loc[movieRating['count']>100].head(10).iterrows():
    printMovieListing(i, movie)

########################################################
# Begin Phase 3
########################################################

# Create a user likes numpy ndarray so we can use Jaccard Similarity
# A user "likes" a movie if they rated it a 4 or 5
# Create a numpy ndarray of zeros with dimensions of max user id + 1 and max movie + 1 (because we'll use them as 1 indexed not zero indexed)

# Find the max movie ID + 1
maxMovie = len(movieNames) + 1

# Find the max user Id + 1
maxUser = len(movieData.user.unique()) + 1

# Create an array of 0s which will fill in with 1s when a user likes a movie
userLikes = np.zeros((maxUser, maxMovie))

# Go through all the rows of the movie data.
# If the user rated a movie as 4 or 5 set userLikes to 1 for that user and movie
# Note: You'll need a for loop and an if statement
for index, rate in movieData.iterrows():
    if rate['rating'] > 3:
        userLikes[rate['user'], rate['movie']] = True



########################################################
# At this point, go back up to the top and fill in the
# functions up there
########################################################

# First sample user
# User Similiarity: 0.133333333333
iLike = [655, 315, 66, 96, 194, 172]
processLikes(iLike)

# What if it's an exact match? We should return the next closest match
# Second sample case
# User Similiarity: 0.172413793103
iLike = [ 79,  96,  98, 168, 173, 176,194, 318, 357, 427, 603]
processLikes(iLike)

# What if we've seen all the movies they liked?
# Third sample case
# User Similiarity: 0.170731707317
iLike = [ 79,  96,  98, 168, 173, 176,194, 318, 357, 427, 603, 1]
processLikes(iLike)

# If your code completes the above recommendations properly, you're ready for the last part,
# allow the user to select any number of movies that they like and then give them recommendations.
# Note: I recommend having them select movies by ID since the titles are really long.
# You can just assume they have a list of movies somewhere so they already know what numbers to type in.
# If you'd like to give them options though, that would be a cool bonus project if you finish early.

