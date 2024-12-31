public class RegionPool
{
    public Random random;
    public int totalClasses;
    public int totalDimensions;
    public int dimensionsPerRegion;
    public int[] dimensionHopper;
    public HashSet<float>[] dimensionValues;
    public int defaultVote;
    public List<Region> regions;

    public RegionPool(int totalClasses, int totalDimensions, int dimensionsPerRegion, List<Sample> samples)
    {
        this.random = new Random();
        this.totalClasses = totalClasses;
        this.totalDimensions = totalDimensions;
        this.dimensionsPerRegion = dimensionsPerRegion;
        this.dimensionHopper = Enumerable.Range(0, totalDimensions).ToArray();
        this.regions = new List<Region>();

        this.dimensionValues = new HashSet<float>[totalDimensions];
        for (int i = 0; i < totalDimensions; i++)
        {
            dimensionValues[i] = new HashSet<float>();
            foreach(Sample sample in samples)
            {
                dimensionValues[i].Add(sample.input[i]);
            }
        }

        // determine default vote as the mode sample
        int[] defaultVotes = new int[totalClasses];
        foreach(Sample sample in samples)
        {
            defaultVotes[sample.output]++;
        }
        this.defaultVote = -1;
        int defaultVoteCount = -1;
        for (int i = 0; i < totalClasses; i++)
        {
            if (defaultVotes[i] > defaultVoteCount)
            {
                this.defaultVote = i;
                defaultVoteCount = defaultVotes[i];
            }
        }
    }

    public void AddRegion(List<Sample> samples)
    {
        int[] dimensions = dimensionHopper.OrderBy(x => random.Next()).Take(dimensionsPerRegion).ToArray();
        float[] mins = new float[dimensionsPerRegion];
        float[] maxs = new float[dimensionsPerRegion];
        for (int i = 0; i < dimensionsPerRegion; i++)
        {
            int dimensionIndex = dimensions[i];
            HashSet<float> dimensionValueSet = dimensionValues[dimensionIndex];
            float a = dimensionValueSet.ElementAt(random.Next(dimensionValueSet.Count));
            float b = dimensionValueSet.ElementAt(random.Next(dimensionValueSet.Count));
            if (a < b)
            {
                mins[i] = a;
                maxs[i] = b;
            }
            else
            {
                mins[i] = b;
                maxs[i] = a;
            }
        }
        int[] votes = new int[totalClasses];
        foreach(Sample sample in samples)
        {
            bool match = true;
            for (int i = 0; i < dimensionsPerRegion; i++)
            {
                int dimensionIndex = dimensions[i];
                if (sample.input[dimensionIndex] < mins[i] || sample.input[dimensionIndex] > maxs[i])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                votes[sample.output]++;
            }
        }
        int vote = -1;
        int voteCount = -1;
        for (int i = 0; i < totalClasses; i++)
        {
            if (votes[i] > voteCount)
            {
                vote = i;
                voteCount = votes[i];
            }
        }
        regions.Add(new Region(dimensions, mins, maxs, vote));
    }

    public int Predict(List<float> input)
    {
        int[] votes = new int[totalClasses];
        foreach (Region region in regions)
        {
            int regionPrediction = region.Predict(input);
            if (regionPrediction != -1)
            {
                votes[regionPrediction]++;
            }
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
        if (predictionCount == 0)
        {
            prediction = defaultVote;
        }
        return prediction;
    }
}

public class Region
{
    public int[] dimensions;
    public float[] min;
    public float[] max;
    public int vote;

    public Region(int[] dimensions, float[] min, float[] max, int vote)
    {
        this.dimensions = dimensions;
        this.min = min;
        this.max = max;
        this.vote = vote;
    }

    public int Predict(List<float> input)
    {
        for (int i = 0; i < dimensions.Length; i++)
        {
            int dimensionIndex = dimensions[i];
            if (input[dimensionIndex] < min[i] || input[dimensionIndex] > max[i])
            {
                return -1;
            }
        }
        return vote;
    }
}