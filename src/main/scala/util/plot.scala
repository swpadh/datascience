package util

import java.awt.Color

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import breeze.plot.Figure
import breeze.plot.scatter

object Plot {

  def scatterPlot(data: RDD[Vector], yindex: Int, ytext: String, xtext: String) =
    {
      val dataY: ArrayBuffer[Double] = ArrayBuffer[Double]()
      val featureData = data.collect().foreach { t => dataY += t.apply(yindex) }
      val dataX: ArrayBuffer[Double] = ArrayBuffer[Double]()
      for (i <- 0 until dataY.size) {
        dataX += i
      }
      val fig = Figure()

      val plt = fig.subplot(0)

      plt += scatter(dataX.toArray, dataY.toArray, { (_: Int) => 10.0 }, { (_: Int) => Color.GREEN })
      plt.xlabel = xtext
      plt.ylabel = ytext
    }
  def jscatterPlot(data: RDD[Vector], yindex: Int, ytext: String, xtext: String) {

    import java.awt._
    import java.awt.geom._
    import org.jfree.chart._
    import org.jfree.chart.plot._
    import org.jfree.data.xy._

    val dataY: ArrayBuffer[Double] = ArrayBuffer[Double]()
    val featureData = data.collect().foreach { t => dataY += t.apply(yindex) }
    val dataX: ArrayBuffer[Double] = ArrayBuffer[Double]()
    for (i <- 0 until dataY.size) {
      dataX += i
    }
    def createDataset(): DefaultXYDataset = {
      val dataset = new DefaultXYDataset
      dataset.addSeries("Series 1", Array(dataY.toArray, dataX.toArray))
      dataset
    }
    def createChart(dataset: XYDataset): JFreeChart = {
      val chart: JFreeChart = ChartFactory.createScatterPlot(
        "Volume", ytext, xtext, dataset,
        PlotOrientation.HORIZONTAL, true, true, false)
      val circle: Shape = new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0)
      val plot: XYPlot = chart.getXYPlot()
      plot.getRenderer().setSeriesPaint(0, Color.blue)
      plot.getRenderer().setSeriesShape(0, circle)
      chart
    }

    val frame = new ChartFrame(
      "Volume", createChart(createDataset()))

    frame.pack()
    frame.setVisible(true)
  }

  def regressionPlot(data: RDD[LabeledPoint], xindex: Int, xtext: String, ytext: String, lrModel: LinearRegressionModel) {

    import java.awt._
    import java.awt.geom._
    import org.jfree.chart._
    import org.jfree.chart.plot._
    import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
    import org.jfree.data.function._
    import org.jfree.data.general.DatasetUtilities
    import org.jfree.data.xy._

    val dataY: ArrayBuffer[Double] = ArrayBuffer[Double]()
    val dataX: ArrayBuffer[Double] = ArrayBuffer[Double]()
    val featureData = data.collect().foreach { t =>
      dataX += t.features.apply(xindex)
      dataY += t.label
    }
    def createDataset(): DefaultXYDataset = {
      val dataset = new DefaultXYDataset
      dataset.addSeries("Median Housing Value", Array(dataY.toArray, dataX.toArray))
      dataset
    }

    def createChart(dataset: XYDataset): JFreeChart = {
      val chart: JFreeChart = ChartFactory.createScatterPlot(
        "Median Housing Value", ytext, xtext, dataset,
        PlotOrientation.HORIZONTAL, true, true, false)
      val circle: Shape = new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0)
      val plot: XYPlot = chart.getXYPlot()
      plot.getRenderer().setSeriesPaint(0, Color.blue)
      plot.getRenderer().setSeriesShape(0, circle)
      chart
    }
    def createRegressionLine(chart: JFreeChart): JFreeChart = {
      val linefunction2d: LineFunction2D = new LineFunction2D(
        lrModel.intercept, lrModel.coefficients.apply(xindex))
      val dataset: XYDataset = DatasetUtilities.sampleFunction2D(linefunction2d,
        0, dataX.max, dataY.size, "Fitted Regression Line")

      val xyplot: XYPlot = chart.getXYPlot()
      xyplot.setDataset(1, dataset)
      val xylineandshaperenderer: XYLineAndShapeRenderer = new XYLineAndShapeRenderer(
        true, false)
      xylineandshaperenderer.setSeriesPaint(0, Color.YELLOW)
      xyplot.setRenderer(1, xylineandshaperenderer)
      chart
    }
    val frame = new ChartFrame("Volume", createRegressionLine(createChart(createDataset)))
    frame.pack()
    frame.setVisible(true)
  }
}