public class PartitionPool
{
    public Random random;
    public int totalClasses;
    public int totalDimensions;
    public int dimensionCount;
    public int partitionCount;
    public int[] dimensionHopper;
    public List<int[]> dimensions;
    public List<float[]> values;
    public List<bool[]> directions;
    public List<int> matchVotes;
    public List<int> complimentVotes;

    public PartitionPool(int totalClasses, int totalDimensions, int dimensionCount)
    {
        this.random = new Random();
        this.totalClasses = totalClasses;
        this.totalDimensions = totalDimensions;
        this.dimensionCount = dimensionCount;
        this.partitionCount = 0;
        this.dimensionHopper = Enumerable.Range(0, totalDimensions).ToArray();
        this.dimensions = new List<int[]>();
        this.values = new List<float[]>();
        this.directions = new List<bool[]>();
        this.matchVotes = new List<int>();
        this.complimentVotes = new List<int>();
    }

    public void AddPartition(List<Sample> samples)
    {
        int[] partitionDimensions = dimensionHopper.OrderBy(x => random.Next()).Take(dimensionCount).ToArray();
        float[] partitionValues = new float[dimensionCount];
        for (int i = 0; i < dimensionCount; i++)
        {
            Sample randomSample = samples[random.Next(samples.Count)];
            int dimensionIndex = partitionDimensions[i];
            partitionValues[i] = randomSample.input[dimensionIndex];
        }
        bool[] partitionDirections = new bool[dimensionCount];
        for (int i = 0; i < dimensionCount; i++)
        {
            partitionDirections[i] = random.Next(2) == 0;
        }
        int[] partitionMatchVotes = new int[totalClasses];
        int[] partitionComplimentVotes = new int[totalClasses];
        foreach(Sample sample in samples)
        {
            bool match = true;
            for (int i = 0; i < dimensionCount; i++)
            {
                int dimensionIndex = partitionDimensions[i];
                if (partitionDirections[i] && sample.input[dimensionIndex] <= partitionValues[i])
                {
                    match = false;
                    break;
                }
                if (!partitionDirections[i] && sample.input[dimensionIndex] >= partitionValues[i])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                partitionMatchVotes[sample.output]++;
            }
            else
            {
                partitionComplimentVotes[sample.output]++;
            }
        }
        int partitionMatchVote = -1;
        int partitionMatchVoteCount = -1;
        int partitionComplimentVote = -1;
        int partitionComplimentVoteCount = -1;
        for (int i = 0; i < totalClasses; i++)
        {
            if (partitionMatchVotes[i] > partitionMatchVoteCount)
            {
                partitionMatchVote = i;
                partitionMatchVoteCount = partitionMatchVotes[i];
            }
            if (partitionComplimentVotes[i] > partitionComplimentVoteCount)
            {
                partitionComplimentVote = i;
                partitionComplimentVoteCount = partitionComplimentVotes[i];
            }
        }
        partitionCount++;
        dimensions.Add(partitionDimensions);
        values.Add(partitionValues);
        directions.Add(partitionDirections);
        matchVotes.Add(partitionMatchVote);
        complimentVotes.Add(partitionComplimentVote);
    }

    public int Predict(List<float> input)
    {
        int[] votes = new int[totalClasses];
        for (int partitionIndex = 0; partitionIndex < partitionCount; partitionIndex++)
        {
            bool match = true;
            for (int i = 0; i < dimensionCount; i++)
            {
                int dimensionIndex = dimensions[partitionIndex][i];
                if (directions[partitionIndex][i] && input[dimensionIndex] <= values[partitionIndex][i])
                {
                    match = false;
                    break;
                }
                if (!directions[partitionIndex][i] && input[dimensionIndex] >= values[partitionIndex][i])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                votes[matchVotes[partitionIndex]]++;
            }
            else
            {
                votes[complimentVotes[partitionIndex]]++;
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
        return prediction;
    }
}