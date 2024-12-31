public class HyperPlanePool
{
    public Random random;
    public int totalClasses;
    public int totalDimensions;
    public int dimensionsPerPlane;
    public List<HyperPlane> hyperPlanes;

    public HyperPlanePool(int totalClasses, int totalDimensions, int dimensionsPerPlane, List<Sample> samples)
    {
        this.random = new Random();
        this.totalClasses = totalClasses;
        this.totalDimensions = totalDimensions;
        this.dimensionsPerPlane = dimensionsPerPlane;
        this.hyperPlanes = new List<HyperPlane>();
    }

    public void AddHyperPlane(List<Sample> samples)
    {
        for (; ; )
        {
            int[] dimensions = Enumerable.Range(0, totalDimensions).OrderBy(x => random.Next()).Take(dimensionsPerPlane).ToArray();
            Sample normalSample = samples[random.Next(samples.Count)];
            Sample pointOnPlaneSample = samples[random.Next(samples.Count)];
            List<float> normal = dimensions.Select(x => normalSample.input[x]).ToList();
            List<float> pointOnPlane = dimensions.Select(x => pointOnPlaneSample.input[x]).ToList();
            if (Enumerable.SequenceEqual(normal, pointOnPlane))
            {
                continue;
            }
            HyperPlane hyperPlane = new HyperPlane(dimensions, normal, pointOnPlane, totalClasses, samples);
            hyperPlanes.Add(hyperPlane);
            break;
        }
    }

    public int Predict(List<float> input)
    {
        int[] votes = new int[totalClasses];
        foreach (HyperPlane hyperPlane in hyperPlanes)
        {
            int hyperPlanePredition = hyperPlane.Predict(input);
            votes[hyperPlanePredition]++;
        }
        int prediction = -1;
        int predictionCount = -1;
        for (int i = 0; i < totalClasses; i++)
        {
            if (votes[i] > predictionCount)
            {
                prediction = i;
                predictionCount = votes[i];
            }
        }
        return prediction;
    }
}

public class HyperPlane
{
    public int[] dimensions;
    public List<float> normal;
    public float b;
    public int voteOver;
    public int voteUnder;

    public HyperPlane(int[] dimensions, List<float> normal, List<float> pointOnPlane, int totalClassCount, List<Sample> samples)
    {
        this.dimensions = dimensions;
        this.normal = normal;
        this.b = -Dot(normal, pointOnPlane);
        int[] votesOver = new int[totalClassCount];
        int[] votesUnder = new int[totalClassCount];
        foreach(Sample sample in samples)
        {
            if (Dot(normal, sample.input) + b > 0)
            {
                votesOver[sample.output]++;
            }
            else
            {
                votesUnder[sample.output]++;
            }
        }
        this.voteOver = -1;
        this.voteUnder = -1;
        int voteOverCount = -1;
        int voteUnderCount = -1;
        for (int i = 0; i < totalClassCount; i++)
        {
            if (votesOver[i] > voteOverCount)
            {
                this.voteOver = i;
                voteOverCount = votesOver[i];
            }
            if (votesUnder[i] > voteUnderCount)
            {
                this.voteUnder = i;
                voteUnderCount = votesUnder[i];
            }
        }
    }

    public float Dot(List<float> vectorA, List<float> vectorB)
    {
        float result = 0;
        for (int i = 0; i < vectorA.Count; i++)
        {
            result += vectorA[i] * vectorB[i];
        }
        return result;
    }

    public int Predict(List<float> input)
    {
        List<float> dimensionalInput = dimensions.Select(x => input[x]).ToList();
        if (Dot(normal, dimensionalInput) + b > 0)
        {
            return voteOver;
        }
        else
        {
            return voteUnder;
        }
    }
}