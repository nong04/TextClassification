using Microsoft.ML.Data;
using CsvHelper.Configuration;
using System.Text.RegularExpressions;

namespace ReviewClassification
{
    public class Review
    {
        [LoadColumn(0)]  // Replace 0 with the actual column index for ReviewText in your CSV file
        public string ReviewText { get; set; }

        [LoadColumn(1)]  // Replace 1 with the actual column index for Rating in your CSV file
        public float Rating { get; set; }

        [LoadColumn(2)]  // Replace 2 with the actual column index for Sentiment in your CSV file
        public string Sentiment { get; set; }
        public Review() { }

        public Review(string ReviewText)
        {
            this.ReviewText = ReviewText;
        }

        public Review(string ReviewText, float Rating) 
        {
            this.ReviewText = ReviewText;
            this.Rating = Rating;
        }

        public Review(string ReviewText, float Rating, string Sentiment) 
        {
            this.ReviewText = ReviewText;
            this.Rating = Rating;
            this.Sentiment = Sentiment;
        }
    }

    public class ReviewMap : ClassMap<Review>
    {
        public ReviewMap()
        {
            Map(m => m.ReviewText).Name("ReviewText");
            Map(m => m.Rating).Name("Rating");
            Map(m => m.Sentiment).Name("Sentiment");
        }
    }

    public class ReviewPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedSentiment { get; set; }
        public float[] Score { get; set; }
    }
}