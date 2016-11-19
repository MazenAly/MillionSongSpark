package se.kth.spark.lab1.task7
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

object Main {
  def main(args: Array[String]) {
   // val conf = new SparkConf().setAppName("lab1").setMaster("local")
   val conf = new SparkConf().setAppName("lab2")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

   // val filePath = "src/main/resources/millionsong.txt"
    val filePath = "hdfs://10.0.104.163:8020/Projects/MazenML/Resources/all.txt"
    //val obsDF: DataFrame = sc.textFile(filePath).map(x => x.split(',')).map(parts => (parts(0).toDouble  , Vectors.dense( parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))).toDF("label", "features")
    
   val obsDF: DataFrame =  sc.textFile(filePath).toDF("raw")

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("tokens")
      .setPattern("[\\d\\.-]+")
      .setGaps(false)

    val arr2Vect = new Array2Vector()
    .setInputCol("tokens")
    .setOutputCol("tokens_vector")

    val lSlicer =  new VectorSlicer().setInputCol("tokens_vector").setOutputCol("label_1")
    .setIndices(Array(0))
    
    //var x = Double.parseDouble("5.5")
    //val v2d = new Vector2DoubleUDF(x=> x(0).intValue().toDouble)
    val v2d = new Vector2DoubleUDF(x=> x(0).toDouble)
    .setInputCol("label_1")
    .setOutputCol("label_d")
   
    val lShifter = new DoubleUDF(x => (x-1922.0))
    .setInputCol("label_d")
    .setOutputCol("label")
    
   
    val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
    .setIndices(Array(1,2,3,4,5,6,7,8,9,10,11,12 , 13))
    
   // best max iter 7
// best reg param 0.001 
    
    val myLR  = new LinearRegression().setMaxIter(7).setRegParam(0.001).setElasticNetParam(0.1)
   
    val lrStage = 6
    
   var pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect , lSlicer,v2d , lShifter, fSlicer, myLR))

   var pipelineModel = (pipeline: Pipeline).fit(obsDF)
    
   val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

   val trainingSummary = lrModel.summary
   //RMSE: 17.616138636430982
   println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
   sc.stop
  }
}