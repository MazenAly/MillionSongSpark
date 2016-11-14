package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sc.textFile(filePath).map(x => x.split(',')).map(parts => (parts(0).toDouble  , Vectors.dense( parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))).toDF("label", "features")
    
    
    
     obsDF.show(5)
    
    val myLR  = new LinearRegression().setMaxIter(10).setRegParam(0.3)
  
  
  
    val lrStage = 0
    //val params = ParamMap(myLR.maxIter -> 10 , myLR.regParam -> 0.3)
    
    val pipeline = new Pipeline().setStages(Array( myLR))
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]


    //print rmse of our model
    //do prediction - print first k
    
    // Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show(5)
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

    
    
    
  }
}