package se.kth.spark.lab1.task4
import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.PolynomialExpansion

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
   val obsDF: DataFrame =  sc.textFile(filePath).toDF("raw")

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("tokens")
      .setPattern(",")

    val arr2Vect = new Array2Vector()
    .setInputCol("tokens")
    .setOutputCol("tokens_vector")

    val lSlicer =  new VectorSlicer().setInputCol("tokens_vector").setOutputCol("label_1")
    .setIndices(Array(0))
    

    val v2d = new Vector2DoubleUDF(x=> x(0).toDouble)
    .setInputCol("label_1")
    .setOutputCol("label_d")
   
    val lShifter = new DoubleUDF(x => (x-1922.0))
    .setInputCol("label_d")
    .setOutputCol("label")
    
   
    val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
    .setIndices(Array(1,2,3))
    
    val myLR  = new LinearRegression().setElasticNetParam(0.1)
   
    val lrStage = 6
    
   var pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect , lSlicer,v2d , lShifter, fSlicer, myLR))
    
val paramGrid = new ParamGridBuilder()
   .addGrid(myLR.maxIter , Array(1,5,7 ,10 ,15,20,30)) // 
   .addGrid(myLR.regParam , Array(0.001, 0.01 ,0.1, 0.5 , 1 , 2)) //
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new RegressionEvaluator )
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(2)  

   val cvModel = cv.fit(obsDF)   


val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]

   println(lrModel.getMaxIter)   
   println(lrModel.getRegParam ) 
   
// best max iter 7
// best reg param 0.001
// RMSE: 17.349253011499027
val trainingSummary = lrModel.summary
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

  }
}