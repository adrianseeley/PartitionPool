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

    public static float Fitness(PartitionPool partitionPool, List<Sample> testSamples)
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


        using TextWriter tw = new StreamWriter("results.csv", false);
        tw.WriteLine("dimensionCount,partitionCount,fitness");

        for (int dimensionCount = 1; dimensionCount <= totalDimensions; dimensionCount++)
        {
            PartitionPool partitionPool = new PartitionPool(totalClasses, totalDimensions, dimensionCount);
            for (int partitionCount = 1; partitionCount < 1000; partitionCount++)
            {
                partitionPool.AddPartition(mnistTrain);
                float fitness = Fitness(partitionPool, mnistTest);
                tw.WriteLine($"{dimensionCount},{partitionCount},{fitness}");
                Console.Write($"\rd: {dimensionCount}, p: {partitionCount}, f: {fitness}            ");
            }
        }
    }
}

