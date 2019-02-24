import boofcv.abst.feature.dense.DescribeImageDense;
import boofcv.alg.bow.ClusterVisualWords;
import boofcv.alg.bow.LearnSceneFromFiles;
import boofcv.alg.scene.ClassifierKNearestNeighborsBow;
import boofcv.alg.scene.FeatureToWordHistogram_F64;
import boofcv.alg.scene.HistogramScene;
import boofcv.factory.feature.dense.*;
import boofcv.gui.image.ShowImages;
import boofcv.gui.learning.ConfusionMatrixPanel;
import boofcv.io.UtilIO;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.feature.TupleDesc_F64;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageType;
import boofcv.struct.learning.Confusion;
import deepboof.io.DeepBoofDataBaseOps;
import org.ddogleg.clustering.AssignCluster;
import org.ddogleg.clustering.ComputeClusters;
import org.ddogleg.clustering.FactoryClustering;
import org.ddogleg.nn.FactoryNearestNeighbor;
import org.ddogleg.nn.NearestNeighbor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Training Steps:
 * <ol>
 * <li>Compute dense SIFT features across the training data set.</li>
 * <li>Cluster using k-means to create words.</li>
 * <li>For each image compute the histogram of words found in the image</li>
 * <li>Save word histograms and image scene labels in a classifier</li>
 * </ol>
 *
 * Testing Steps:
 * <ol>
 * <li>For each image in the testing data set compute its histogram</li>
 * <li>Look up the k-nearest-neighbors for that histogram</li>
 * <li>Classify an image by selecting the scene type with the most neighbors</li>
 * </ol>
 */

public class Demo extends LearnSceneFromFiles {

	// Tuning parameters
	public static int NUMBER_OF_WORDS = 150;
	public static int NUM_NEIGHBORS = 10;
	public static int MAX_KNN_ITERATIONS = 100;

	// Files intermediate results are stored in
	public static final String CLUSTER_FILE_NAME = "clusters.obj";
	public static final String HISTOGRAM_FILE_NAME = "histograms.obj";
	public static boolean HISTOGRAM_HARD = true;


	// Algorithms
	ClusterVisualWords cluster;
	DescribeImageDense<GrayU8,TupleDesc_F64> describeImage;
	NearestNeighbor<HistogramScene> nn;

	ClassifierKNearestNeighborsBow<GrayU8,TupleDesc_F64> classifier;

	public Demo(final DescribeImageDense<GrayU8, TupleDesc_F64> describeImage,
								   ComputeClusters<double[]> clusterer,
								   NearestNeighbor<HistogramScene> nn) {
		this.describeImage = describeImage;
		this.cluster = new ClusterVisualWords(clusterer, describeImage.createDescription().size(),0xFEEDBEEF);
		this.nn = nn;
	}

	/**
	 * Process all the data in the training data set to learn the classifications.
	 */
	public void learn() {
		System.out.println("======== Learning Classifier");

		// Either load pre-computed words or compute the words from the training images
		AssignCluster<double[]> assignment;
		if( new File(CLUSTER_FILE_NAME).exists() ) {
			assignment = UtilIO.load(CLUSTER_FILE_NAME);
		} else {
			System.out.println(" Computing clusters");
			assignment = computeClusters();
		}

		// Use these clusters to assign features to words
		FeatureToWordHistogram_F64 featuresToHistogram = new FeatureToWordHistogram_F64(assignment,HISTOGRAM_HARD);

		// Storage for the work histogram in each image in the training set and their label
		List<HistogramScene> memory;

		if( !new File(HISTOGRAM_FILE_NAME).exists() ) {
			System.out.println(" computing histograms");
			memory = computeHistograms(featuresToHistogram);
			UtilIO.save(memory,HISTOGRAM_FILE_NAME);
		}
	}

	/**
	 * Extract dense features across the training set.  Then clusters are found within those features.
	 */
	private AssignCluster<double[]> computeClusters() {
		System.out.println("Image Features");

		// computes features in the training image set
		List<TupleDesc_F64> features = new ArrayList<>();
		for( String scene : train.keySet() ) {
			List<String> imagePaths = train.get(scene);
			System.out.println("   " + scene);

			for( String path : imagePaths ) {
				GrayU8 image = UtilImageIO.loadImage(path, GrayU8.class);
				describeImage.process(image);

				// the descriptions will get recycled on the next call, so create a copy
				for( TupleDesc_F64 d : describeImage.getDescriptions() ) {
					features.add( d.copy() );
				}
			}
		}
		// add the features to the overall list which the clusters will be found inside of
		for (int i = 0; i < features.size(); i++) {
			cluster.addReference(features.get(i));
		}

		System.out.println("Clustering");
		// Find the clusters.  This can take a bit
		cluster.process(NUMBER_OF_WORDS);

		UtilIO.save(cluster.getAssignment(), CLUSTER_FILE_NAME);

		return cluster.getAssignment();
	}

	public void createClassifier() {
		// load results from a file
		List<HistogramScene> memory = UtilIO.load(HISTOGRAM_FILE_NAME);
		AssignCluster<double[]> assignment = UtilIO.load(CLUSTER_FILE_NAME);

		FeatureToWordHistogram_F64 featuresToHistogram = new FeatureToWordHistogram_F64(assignment,HISTOGRAM_HARD);


		// Provide the training results to K-NN and it will preprocess these results for quick lookup later on

		classifier = new ClassifierKNearestNeighborsBow<>(nn, describeImage, featuresToHistogram);
		classifier.setClassificationData(memory, getScenes().size());
		classifier.setNumNeighbors(NUM_NEIGHBORS);
	}

	/**
	 * For all the images in the training data set it computes a {@link HistogramScene}.  That data structure
	 * contains the word histogram and the scene that the histogram belongs to.
	 */
	private List<HistogramScene> computeHistograms(FeatureToWordHistogram_F64 featuresToHistogram ) {

		List<String> scenes = getScenes();

		List<HistogramScene> memory;// Processed results which will be passed into the k-NN algorithm
		memory = new ArrayList<>();

		for( int sceneIndex = 0; sceneIndex < scenes.size(); sceneIndex++ ) {
			String scene = scenes.get(sceneIndex);
			System.out.println("   " + scene);
			List<String> imagePaths = train.get(scene);

			for (String path : imagePaths) {
				GrayU8 image = UtilImageIO.loadImage(path, GrayU8.class);

				// reset before processing a new image
				featuresToHistogram.reset();
				describeImage.process(image);
				for ( TupleDesc_F64 d : describeImage.getDescriptions() ) {
					featuresToHistogram.addFeature(d);
				}
				featuresToHistogram.process();

				// The histogram is already normalized so that it sums up to 1.  This provides invariance
				// against the overall number of features changing.
				double[] histogram = featuresToHistogram.getHistogram();

				// Create the data structure used by the KNN classifier
				HistogramScene imageHist = new HistogramScene(NUMBER_OF_WORDS);
				imageHist.setHistogram(histogram);
				imageHist.type = sceneIndex;

				memory.add(imageHist);
			}
		}
		return memory;
	}

	@Override
	protected int classify(String path) {
		GrayU8 image = UtilImageIO.loadImage(path, GrayU8.class);

		return classifier.classify(image);
	}

	public static void main(String[] args) {

		ConfigDenseSurfStable surf = new ConfigDenseSurfStable(new DenseSampling(8,8));
		ConfigDenseSift sift = new ConfigDenseSift(new DenseSampling(6,6));
		ConfigDenseHoG hog = new ConfigDenseHoG();

		DescribeImageDense<GrayU8,TupleDesc_F64> desc = (DescribeImageDense)
//				FactoryDescribeImageDense.surfStable(surf, GrayU8.class);
				FactoryDescribeImageDense.sift(sift, GrayU8.class);
//				FactoryDescribeImageDense.hog(hog, ImageType.single(GrayU8.class));

		ComputeClusters<double[]> clusterer = FactoryClustering.kMeans_F64(null, MAX_KNN_ITERATIONS, 20, 1e-6);
		clusterer.setVerbose(true);

		NearestNeighbor<HistogramScene> nn = FactoryNearestNeighbor.exhaustive();
		Demo demo = new Demo(desc,clusterer,nn);

		File trainingDir = new File(UtilIO.pathExample("learning/scene/train"));
		File testingDir = new File(UtilIO.pathExample("learning/scene/test"));

		if( !trainingDir.exists() || !testingDir.exists() ) {
			String addressSrc = "http://boofcv.org/notwiki/largefiles/bow_data_v001.zip";
			File dst = new File(trainingDir.getParentFile(),"bow_data_v001.zip");
			try {
				DeepBoofDataBaseOps.download(addressSrc, dst);
				DeepBoofDataBaseOps.decompressZip(dst, dst.getParentFile(), true);
				System.out.println("Download complete!");
			} catch( IOException e ) {
				throw new RuntimeException(e);
			}
		} else {
			System.out.println("Delete and download again if there are file not found errors");
			System.out.println("   "+trainingDir);
			System.out.println("   "+testingDir);
		}

		demo.loadSets(trainingDir, null, testingDir);
		// train the classifier
		demo.learn();
		// now load it for evaluation purposes from the files
		demo.createClassifier();

		// test the classifier on the test set
		Confusion confusion = demo.evaluateTest();
		confusion.getMatrix().print();
		System.out.println("Accuracy = " + confusion.computeAccuracy());

		// Show confusion matrix
		ShowImages.showWindow(new ConfusionMatrixPanel(
				confusion.getMatrix(),demo.getScenes(), 400, true), "Confusion Matrix", true);
	}
}
