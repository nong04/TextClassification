using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using System.Globalization;
using System.Text;
using ReviewClassification;
using CsvHelper;
using CsvHelper.Configuration;
using System.Text.RegularExpressions;
using static TorchSharp.torch.nn;
using System.Collections.Generic;
using NHunspell;
using static TorchSharp.torch.utils;
using static System.Net.Mime.MediaTypeNames;

//Dictionary link:https://github.com/LibreOffice/dictionaries/tree/master/en
//Last updated in project: 2024-12-06
class Program
{
    static void Main(string[] args)
    {
        string savePath = @"..\..\Restaurant Reviews.csv"; // Original file

        string modelPath = @"..\..\TextTransformation_0.7131_0.7027.zip";

        Console.Write("Reading csv to data: ");
        List<Review> data = ReadCSV(savePath);
        Console.WriteLine("Finished.");

        //PrintData(data);

        //ReportDataIssues(data);
        Console.Write("Removing invalid data: ");
        List<Review> data2 = CleanData(data);
        SaveReviewsToCsv(data2, @"..\..\data2.csv");

        Console.Write("Balancing dataset: ");
        List<Review> data3 = BalanceDataset(data2, 2);
        SaveReviewsToCsv(data3, @"..\..\data3.csv");

        Console.WriteLine("Correcting spelling:");
        List<Review> data4 = CorrectSpelling(data3);
        SaveReviewsToCsv(data4, @"..\..\data4.csv");

        Console.Write("Expanding acronyms: ");
        List<Review> data5 = ExpandAcronyms(data4);
        SaveReviewsToCsv(data5, @"..\..\data5.csv");

        Console.Write("Normalizing data: ");
        List<Review> data6 = NormalizeData(data5);
        SaveReviewsToCsv(data6, @"..\..\data6.csv");

        Console.Write("Assigning labels: ");
        List<Review> data7 = AssignLabel(data6);
        SaveReviewsToCsv(data7, @"..\..\data7.csv");

        //Console.WriteLine("Training Data:");
        //TrainData(data);

        //Console.WriteLine("Testing model:");
        //Test(modelPath);
    }

    public static List<Review> ReadCSV(string filePath)
    {
        using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var reader = new StreamReader(fileStream))
        using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            MissingFieldFound = null, // Ignore missing fields if any
            HeaderValidated = null, // Ignore header validation
            BadDataFound = null // Ignore bad data
        }))
        {
            csv.Context.RegisterClassMap<ReviewMap>();

            // Load records, handle empty Sentiment
            var records = csv.GetRecords<Review>().Select(r => new Review
            {
                ReviewText = r.ReviewText,
                Rating = r.Rating,
                Sentiment = string.IsNullOrEmpty(r.Sentiment) ? "Unknown" : r.Sentiment // Default to "Unknown" if empty
            }).ToList();

            return records;
        }
    }

    public static void SaveReviewsToCsv(List<Review> reviews, string filePath)
    {
        using (var writer = new StreamWriter(filePath))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(reviews); // Write the list to the CSV file
        }
        Console.WriteLine($"Data saved to: {filePath}");
    }

    private static void PrintData(List<Review> data)
    {
        foreach (Review obj in data)
        {
            Console.WriteLine(obj.ReviewText);
            Console.WriteLine(obj.Rating);
            Console.WriteLine(obj.Sentiment);
            Console.WriteLine("-------------------------");
        }
    }

    private static void ReportDataIssues(List<Review> data)
    {
        // Check for empty records
        var emptyRecords = data
            .Select((review, index) => new { review, index })
            .Where(record => string.IsNullOrWhiteSpace(record.review.ReviewText))
            .ToList();

        if (emptyRecords.Count > 0)
        {
            Console.WriteLine("Empty records found:");
            foreach (var record in emptyRecords)
            {
                Console.WriteLine($"Empty ReviewText at Position: {record.index}");
            }
            Console.WriteLine($"Total Empty Records: {emptyRecords.Count}\n");
        }
        else
        {
            Console.WriteLine("No empty records found.\n");
        }

        // Check for invalid Rating records
        var invalidRatingRecords = data
            .Select((review, index) => new { review, index })
            .Where(record => record.review.Rating <= 0)
            .ToList();

        if (invalidRatingRecords.Count > 0)
        {
            Console.WriteLine("Records with invalid Rating found:");
            foreach (var record in invalidRatingRecords)
            {
                Console.WriteLine($"Invalid Rating ({record.review.Rating}) at Position: {record.index}");
            }
            Console.WriteLine($"Total Invalid Rating Records: {invalidRatingRecords.Count}\n");
        }
        else
        {
            Console.WriteLine("No invalid Rating records found.\n");
        }

        // Check for duplicate records
        var duplicateRecords = data
            .Select((review, index) => new { review, index })
            .GroupBy(record => new
            {
                record.review.ReviewText,
                record.review.Rating
            })
            .Where(group => group.Count() > 1)
            .ToList();

        if (duplicateRecords.Count > 0)
        {
            Console.WriteLine("Duplicate records found:");
            foreach (var group in duplicateRecords)
            {
                Console.WriteLine($"Duplicate ReviewText: \"{group.Key.ReviewText}\", Rating: {group.Key.Rating}");
                Console.WriteLine("Positions: " + string.Join(", ", group.Select(x => x.index)));
                Console.WriteLine($"Count: {group.Count()}");
                Console.WriteLine();
            }
        }
        else
        {
            Console.WriteLine("No duplicate records found.\n");
        }
    }

    private static List<Review> CleanData(List<Review> data)
    {
        // Remove records with empty ReviewText or invalid Rating
        var validRecords = data
            .Where(review => !string.IsNullOrWhiteSpace(review.ReviewText) && review.Rating > 0)
            .ToList();

        // Remove duplicate records
        var uniqueRecords = validRecords
            .GroupBy(record => new { record.ReviewText, record.Rating })
            .Select(group => group.First())
            .ToList();

        return uniqueRecords;
    }

    public static List<Review> BalanceDataset(List<Review> reviews, int parts)
    {
        double step = 5.0 / parts;
        var groups = new List<List<Review>>();

        for (int i = 0; i < parts; i++)
        {
            double lowerBound = i * step;
            double upperBound = (i + 1) * step;
            groups.Add(reviews.Where(r => r.Rating > lowerBound && r.Rating <= upperBound).ToList());
        }
        int minCount = groups.Min(g => g.Count);
        Random random = new Random();
        for (int i = 0; i < groups.Count; i++)
        {
            if (groups[i].Count > minCount)
            {
                groups[i] = groups[i].OrderBy(_ => random.Next()).Take(minCount).ToList();
            }
        }
        return groups.SelectMany(g => g).ToList();
    }

    public static List<Review> ExpandAcronyms(List<Review> data)
    {
        List<Review> data2 = new List<Review>();
        string pattern = @"\b(?i)('m|'re|'s|'d|'ll|'ve|n't|can't|won't|isn't|wasn't|aren't|don't|doesn't|haven't|hadn't|didn't|couldn't)\b";

        foreach (Review rv in data)
        {
            string x = rv.ReviewText;
            x = x.Replace("\u2019", "'");
            string y = Regex.Replace(x, pattern, match =>
            {
                switch (match.Value.ToLower())
                {
                    case "'m": return " am";
                    case "'re": return " are";
                    case "'s": return " is";
                    case "'d": return " would";
                    case "'ll": return " will";
                    case "'ve": return " have";
                    case "wouldn't": return "would not";
                    case "shouldn't": return "should not";
                    case "can't": return "can not";
                    case "won't": return "will not";
                    case "isn't": return "is not";
                    case "wasn't": return "was not";
                    case "aren't": return "are not";
                    case "don't": return "do not";
                    case "doesn't": return "does not";
                    case "haven't": return "have not";
                    case "hadn't": return "had not";
                    case "didn't": return "did not";
                    case "couldn't": return "could not";

                    default: return match.Value;
                }
            }).Trim();
            var z = Regex.Replace(y, @"[^a-zA-Z0-9\s.,!?'-]", "");

            Review review = new Review(z, rv.Rating, rv.Sentiment);
            data2.Add(review);
        }
        return data2;
    }

    private static List<Review> NormalizeData(List<Review> data)
    {
        var mlContext = new MLContext();

        var pipeline = mlContext.Transforms.Text.NormalizeText(
            outputColumnName: nameof(Review.ReviewText), inputColumnName: nameof(Review.ReviewText),
            keepDiacritics: false, keepPunctuations: false, keepNumbers: true);

        var reviews = mlContext.Data.LoadFromEnumerable(data);

        var normalizedData = pipeline.Fit(reviews).Transform(reviews);

        var ret = mlContext.Data.CreateEnumerable<Review>(normalizedData, reuseRowObject: false).ToList();
        return ret;
    }

    private static string CorrectText(Hunspell hunspell, string text)
    {
        var words = text.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            if (!hunspell.Spell(words[i]))
            {
                var suggestions = hunspell.Suggest(words[i]);
                if (suggestions.Count > 0)
                {
                    words[i] = suggestions[0];
                }
            }
        }
        return string.Join(" ", words);
    }

    public static List<Review> CorrectSpelling(List<Review> reviews)
    {
        using (Hunspell hunspell = new Hunspell("en_US.aff", "en_US.dic"))
        {
            int i = 1;
            foreach (var review in reviews)
            {
                Console.WriteLine($"Correcting spelling: {i++}/{reviews.Count}");
                review.ReviewText = CorrectText(hunspell, review.ReviewText);
            }
        }
        return reviews;
    }

    private static List<Review> AssignLabel(List<Review> data)
    {
        List<Review> ret = new List<Review>();
        foreach (Review review in data)
        {
            string Sentiment;
            if (review.Rating < 2.5) Sentiment = "Negative";
            //else if (review.Rating >= 3 && review.Rating <= 4) Sentiment = "Neutral";
            else if (review.Rating >= 2.5) Sentiment = "Positive";
            else Sentiment = "Unknown";

            Review rv = new Review(review.ReviewText, review.Rating, Sentiment);
            ret.Add(rv);
        }
        return ret;
    }

    public static void TrainData(List<Review> reviews)
    {
        // Initialize MLContext
        MLContext mlContext = new()
        {
            GpuDeviceId = 0,
            FallbackToCpu = true
        };

        var data = mlContext.Data.LoadFromEnumerable(reviews);

        //foreach (var row in df.Rows)
        //{
        //    Console.WriteLine(string.Join("\t", row));
        //}

        // Split the data into train and test sets.
        var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        //Define your training pipeline
        Console.Write("Define training pipeline: ");
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", nameof(Review.Sentiment))
            .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "ReviewText"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        Console.WriteLine("Finished.");

        // Train the model
        Console.Write("Train the model: ");
        var model = pipeline.Fit(trainData);
        Console.WriteLine("Finished.");

        // Use the model to make predictions
        var predictionIDV = model.Transform(testData);

        // Evaluate the model
        Console.WriteLine("Evaluate the model:");
        var evaluationMetrics = mlContext.MulticlassClassification.Evaluate(predictionIDV);
        
        Console.WriteLine($"MacroAccuracy: {evaluationMetrics.MacroAccuracy:F4}");
        Console.WriteLine($"MicroAccuracy: {evaluationMetrics.MicroAccuracy:F4}");
        Console.WriteLine($"LogLoss: {evaluationMetrics.LogLoss:F4}");
        Console.WriteLine(evaluationMetrics.ConfusionMatrix.GetFormattedConfusionTable());

        // Save the model
        string date = DateTime.Now.ToString("dd.MM.yyyy_HH.mm.ss");
        string modelPath = @"..\..\TextClassification.zip";
        string saveFilePath = modelPath.Replace(".zip", $"_{date}.zip");
        mlContext.Model.Save(model, trainData.Schema, saveFilePath);
        Console.WriteLine($"Model saved to: {saveFilePath}");

        // Save the evaluated informations
        string logFilePath = saveFilePath.Replace(".zip", "_log.txt");
        using (StreamWriter writer = new StreamWriter(logFilePath))
        {
            writer.WriteLine($"MacroAccuracy: {evaluationMetrics.MacroAccuracy:F4}");
            writer.WriteLine($"MicroAccuracy: {evaluationMetrics.MicroAccuracy:F4}");
            writer.WriteLine($"TopKPredictionCount: {evaluationMetrics.TopKPredictionCount}");
            writer.WriteLine($"TopKAccuracy: {evaluationMetrics.TopKAccuracy:F4}");
            for (int i = 0; i < evaluationMetrics.TopKPredictionCount; i++)
            {
                writer.WriteLine($"TopKAccuracy for K{i}: {evaluationMetrics.TopKAccuracyForAllK[i]:F4}");
            }
            writer.WriteLine($"LogLoss: {evaluationMetrics.LogLoss:F4}");
            writer.WriteLine($"LogLossReduction: {evaluationMetrics.LogLossReduction:F4}");
            for (int i = 0; i < evaluationMetrics.PerClassLogLoss.Count; i++)
            {
                writer.WriteLine($"LogLoss in class {i}: {evaluationMetrics.PerClassLogLoss[i]:F4}");
            }
            writer.WriteLine(evaluationMetrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
        Console.WriteLine($"Log saved to: {logFilePath}");
    }

    private static string PredictFromModel(string modelPath, string review)
    {
        var mlContext = new MLContext();
        DataViewSchema modelSchema;
        var loadedModel = mlContext.Model.Load(modelPath, out modelSchema);

        // Prediction example with loaded model
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Review, ReviewPrediction>(loadedModel);
        var sampleReview = new Review { ReviewText = review };
        var prediction = predictionEngine.Predict(sampleReview);

        return prediction.PredictedSentiment;
    }

    private static void Test(string modelPath)
    {
        var feedbackSamples = new[]
{
    "The pumpkin spice latte was delightful! I can't get enough of it.",
    "I visited Starbucks, and the service was incredibly slow. I was very disappointed.",
    "The coffee was great, but the atmosphere was a bit too loud for my liking.",
    "I love the seasonal flavors! They always bring something new and exciting.",
    "My experience was average; the coffee was fine, but nothing extraordinary.",
    "The staff was friendly, but my order was wrong. I had to wait again for it to be corrected.",
    "I enjoy sitting in the cozy corner with a book and a hot drink. A perfect afternoon!",
    "The new holiday drinks are fantastic! I can’t wait to try them all.",
    "Unfortunately, my last visit was a letdown. The mocha was too sweet and left a bad aftertaste.",
    "I had a wonderful time chatting with friends over coffee. The vibe is always welcoming.",
    "The location is great, but parking is a hassle. It often discourages me from visiting.",
    "The iced coffee was refreshing, especially on a hot day. I'll be back for more!",
    "I don't usually drink coffee, but the teas here are superb! I highly recommend them.",
    "I waited too long in line, and when I got my drink, it was lukewarm.",
    "Starbucks is my go-to place for meetings. It's convenient and the Wi-Fi is reliable.",
    "I've always had good experiences here. The baristas know their stuff!",
    "I tried the new oat milk option, and it was delicious! A great alternative.",
    "I was disappointed with the cleanliness of the shop. It could use some attention.",
    "The loyalty program is fantastic! I love getting rewards for my purchases.",
    "Every time I go, I find something new to love. Highly recommend!",
    "i ordered a grande tea and they used only one tea bag, the same as a tall tea. what is the extra charge for in the grande size? water? come on, give me a break.",
    "i order the same things at many starbucks in california. this is the only starbucks that charges me more for the same product. i always order a grande americano with steamed breve. i am charged 65 cents more. i am told it is everything from the breve to the labor. everywhere else it is $2.55."
};
        for (int i = 0; i < feedbackSamples.Length; i++)
        {
            Console.WriteLine(feedbackSamples[i] + " :   " + PredictFromModel(modelPath, feedbackSamples[i]));
        }
    }
}