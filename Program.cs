public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            List<float> input = new List<float>();
            for (int i = 1; i < parts.Length; i++)
            {
                input.Add(float.Parse(parts[i]));
            }
            samples.Add(new Sample(input, labelInt));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static float Fitness(RegionPool partitionPool, List<Sample> testSamples)
    {
        int correct = 0;
        foreach (Sample testSample in testSamples)
        {
            int prediction = partitionPool.Predict(testSample.input);
            if (prediction == testSample.output)
            {
                correct++;
            }
        }
        return (float)correct / (float)testSamples.Count;
    }

    public static void Main()
    {
        Random random = new Random();
        
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        int totalClasses = 10;
        int totalDimensions = mnistTrain[0].input.Count;

        List<int> redundantDimensions = new List<int>();
        for (int i = 0; i < totalDimensions; i++)
        {
            bool redundant = true;
            for (int j = 1; j < mnistTrain.Count; j++)
            {
                if (mnistTrain[j].input[i] != mnistTrain[0].input[i])
                {
                    redundant = false;
                    break;
                }
            }
            if (redundant)
            {
                redundantDimensions.Add(i);
            }
        }

        using TextWriter tw = new StreamWriter("results.csv", false);
        tw.WriteLine("dimensionPerRegion,regionCount,fitness");
        object writeLock = new object();

        Parallel.For(1, totalDimensions, i =>
        {
            int dimensionsPerRegion = i;
            RegionPool regionPool = new RegionPool(totalClasses, totalDimensions, dimensionsPerRegion, mnistTrain);
            for (int regionCount = 1; regionCount < 1000; regionCount++)
            {
                regionPool.AddRegion(mnistTrain);
                float fitness = Fitness(regionPool, mnistTest);
                lock (writeLock)
                {
                    tw.WriteLine($"{dimensionsPerRegion},{regionCount},{fitness}");
                    Console.Write($"\rd: {dimensionsPerRegion}, r: {regionCount}, f: {fitness}            ");
                }
            }
        });
    }
}

