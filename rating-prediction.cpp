#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <cmath>
using namespace std;

vector<pair<int, int>> testPairs; // Store userId, movieId pairs for test set
unordered_map<int, unordered_map<int, double>> trainData; // Store userId, movieId and rating for train set
unordered_map<int, unordered_map<int, double>> userSimilarities; // Store user-user similarities
unordered_map<int, unordered_map<int, double>> itemSimilarities; // Store item-item similarities

void store_data() {
    string line;
    bool isTest = false; // Determine if we're in the test dataset section

    // Storing the input
    while (getline(cin, line)) {
        if (line == "train dataset") {
            isTest = false;
            continue;
        }
        else if (line == "test dataset") {
            isTest = true;
            continue;
        }
    
        stringstream ss(line);
        int userId, movieId;
        double rating;
        
        // Adding the test set into the testPairs variable
        if (isTest) {
            ss >> userId >> movieId;
            testPairs.emplace_back(userId, movieId);
        }
        else { // Adding the training set into the trainData variable
            ss >> userId >> movieId >> rating;
            trainData[userId][movieId] = rating;
        }
    }
}

double calculateSimilarity(const unordered_map<int, double>& user1, const unordered_map<int, double>& user2) {
    double dotProduct = 0, A_squared = 0, B_squared = 0;

    // Compute the dot product and the A_squared value of the first user's ratings
    for (const auto& item : user1) {
        int movieId = item.first;
        double rating1 = item.second;
        if (user2.count(movieId)) { // Check if the second user rated the same movie
            double rating2 = user2.at(movieId);
            dotProduct += rating1 * rating2;
        }
        A_squared += rating1 * rating1;
    }

    // Compute the B_squared of the second user's ratings
    for (const auto& item : user2) {
        double rating2 = item.second;
        B_squared += rating2 * rating2;
    }

    // Avoid division by zero
    if (A_squared == 0 || B_squared == 0) return 0.0;
    return dotProduct / (sqrt(A_squared) * sqrt(B_squared));
}

void precomputeUserSimilarities() {
    for (const auto& user1 : trainData) {
        int userId1 = user1.first;
        for (const auto& user2 : trainData) {
            int userId2 = user2.first;
            if (userId1 != userId2) {
                double similarity = calculateSimilarity(user1.second, user2.second);
                userSimilarities[userId1][userId2] = similarity;
            }
        }
    }
}

double calculateItemSimilarity(const unordered_map<int, double>& item1, const unordered_map<int, double>& item2) {
    double dotProduct = 0.0, A_squared = 0.0, B_squared = 0.0;

    for (const auto& item : item1) {
        int userId = item.first;
        double rating1 = item.second;
        if (item2.count(userId)) {
            double rating2 = item2.at(userId);
            dotProduct += rating1 * rating2;
        }
        A_squared += rating1 * rating1;
    }

    for (const auto& item : item2) {
        double rating2 = item.second;
        B_squared += rating2 * rating2;
    }

    if (A_squared == 0 || B_squared == 0) return 0.0;
    return dotProduct / (sqrt(A_squared) * sqrt(B_squared));
}

void precomputeItemSimilarities() {
    unordered_map<int, unordered_map<int, double>> itemRatings;

    //  Change trainData rows and columns (transpose) to get the ratings per item
    for (const auto& [userId, ratings] : trainData) {
        for (const auto& [movieId, rating] : ratings) {
            itemRatings[movieId][userId] = rating;
        }
    }

    for (const auto& item1 : itemRatings) {
        int movieId1 = item1.first;
        for (const auto& item2 : itemRatings) {
            int movieId2 = item2.first;
            if (movieId1 != movieId2) {
                double similarity = calculateItemSimilarity(item1.second, item2.second);
                itemSimilarities[movieId1][movieId2] = similarity;
            }
        }
    }
}

double predictRatingUBCF(int userId, int movieId, int k = 5) {
    vector<pair<double, int>> similarities; // Store similarity and user IDs

    // Use precomputed similarities
    for (const auto& [otherUserId, otherRatings] : trainData) {
        if (otherUserId != userId && otherRatings.count(movieId)) {
            double similarity = userSimilarities[userId][otherUserId];
            similarities.push_back({ similarity, otherUserId });
        }
    }

    // Sort the similarities in descending order
    sort(similarities.rbegin(), similarities.rend());

    double weightedSum = 0, similaritySum = 0;
    int count = 0;

    // Use the k most similar users to predict the rating
    for (const auto& sim : similarities) {
        if (count >= k) {
            break;
        } // Limit to k users
        double similarity = sim.first;
        int otherUserId = sim.second;
        double rating = trainData[otherUserId].at(movieId);
        weightedSum += similarity * rating; // Weighted sum of ratings
        similaritySum += abs(similarity);  // Sum of similarity weights
        count++;
    }

    // Return a default rating of 3.6 if no similar users are found
    if (similaritySum == 0) return 3.6;
    return weightedSum / similaritySum;
}

double predictRatingIBCF(int userId, int movieId, int k = 10) {
    vector<pair<double, int>> similarities;

    for (const auto& [otherMovieId, similarity] : itemSimilarities[movieId]) {
        if (trainData[userId].count(otherMovieId)) {
            similarities.push_back({ similarity, otherMovieId });
        }
    }

    sort(similarities.rbegin(), similarities.rend());

    double weightedSum = 0.0, similaritySum = 0.0;
    int count = 0;

    for (const auto& sim : similarities) {
        if (count >= k) break;
        double similarity = sim.first;
        int similarMovieId = sim.second;
        double rating = trainData[userId].at(similarMovieId);
        weightedSum += similarity * rating;
        similaritySum += abs(similarity);
        count++;
    }

    if (similaritySum == 0) return 3.6;
    return weightedSum / similaritySum;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    store_data();

    // Precompute user-user similarity
    //precomputeUserSimilarities();
    // Precompute item-item similarity
    precomputeItemSimilarities();

    // Predict and output ratings for all test pairs
    for (const auto& [userId, movieId] : testPairs) {
        // Predict using UBCF
        //double predictedRating = predictRatingUBCF(userId, movieId);
        //cout << predictedRating << endl;
        
        // Predict using IBCF
        double predictedRating = predictRatingIBCF(userId, movieId);
        cout << predictedRating << endl;
    }

    return 0;
}
