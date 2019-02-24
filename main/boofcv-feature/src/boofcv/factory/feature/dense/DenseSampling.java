package boofcv.factory.feature.dense;

/**
 * Specifies how the image should be sampled when computing dense features
 *
 * @author Peter Abeles
 */
public class DenseSampling {
	/**
	 * Sample period along x-axis in pixels
	 */
	public double periodX;
	/**
	 * Sample period along y-axis in pixels
	 */
	public double periodY;

	public DenseSampling(double periodX, double periodY) {
		this.periodX = periodX;
		this.periodY = periodY;
	}

	public DenseSampling() {
	}
}
