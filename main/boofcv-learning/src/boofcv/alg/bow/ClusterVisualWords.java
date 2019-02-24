package boofcv.alg.bow;

import boofcv.struct.feature.TupleDesc_F64;
import org.ddogleg.clustering.AssignCluster;
import org.ddogleg.clustering.ComputeClusters;

import java.util.ArrayList;
import java.util.List;

/**
 * Finds clusters of {@link TupleDesc_F64} which can be used to identify frequent features, a.k.a words.
 * Internally it uses {@link org.ddogleg.clustering.ComputeClusters} and simply extracts the inner array
 * from the tuple.
 *
 */
public class ClusterVisualWords {

	// cluster finding algorithm
	ComputeClusters<double[]> computeClusters;

	// inner arrays extracted from the input features
	List<double[]> tuples = new ArrayList<>();

	/**
	 * Constructor which configures the cluster finder.
	 *
	 * @param computeClusters Cluster finding algorithm.
	 * @param featureDOF Number of elements in the feature
	 * @param randomSeed Seed for random number generator
	 */
	public ClusterVisualWords(ComputeClusters<double[]> computeClusters, int featureDOF, long randomSeed) {
		this.computeClusters = computeClusters;

		computeClusters.init(featureDOF,randomSeed);
	}

	/**
	 * Add a feature to the list.
	 *
	 * @param feature image feature. Reference to inner array is saved.
	 */
	public void addReference(TupleDesc_F64 feature) {
		tuples.add(feature.getValue());
	}

	/**
	 * Clusters the list of features into the specified number of words
	 * @param numberOfWords Number of words/clusters it should find
	 */
	public void process( int numberOfWords ) {
		computeClusters.process(tuples,numberOfWords);
	}

	/**
	 * Returns a transform from point to cluster.
	 */
	public AssignCluster<double[]> getAssignment() {
		return computeClusters.getAssignment();
	}

}
