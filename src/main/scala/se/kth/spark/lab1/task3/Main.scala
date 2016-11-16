package se.kth.spark.lab1.task3
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
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //val obsDF: DataFrame = sc.textFile(filePath).map(x => x.split(',')).map(parts => (parts(0).toDouble  , Vectors.dense( parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))).toDF("label", "features")
    
        val obsDF: DataFrame =  sc.textFile(filePath).toDF("raw")

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val regexTokenized = regexTokenizer.transform(obsDF)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    .setInputCol("tokens")
    .setOutputCol("tokens_vector")

    val tokensVectorized = arr2Vect.transform(regexTokenized)
    
    
    //Step4: extract the label(year) into a new column
    val lSlicer =  new VectorSlicer().setInputCol("tokens_vector").setOutputCol("label_1")
    .setIndices(Array(0))
    
    val sliced = lSlicer.transform(tokensVectorized)
    
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(x=> x(0).toDouble)
    .setInputCol("label_1")
    .setOutputCol("label_d")
   
      val label_double = v2d.transform(sliced)    


    val min_label = label_double.select(min($"label_d")).rdd.first.getDouble(0)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    val lShifter = new DoubleUDF(x => (x-min_label))
    .setInputCol("label_d")
    .setOutputCol("label")
    
    
     val lshifted = lShifter.transform(label_double)
    
    
    
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
    .setIndices(Array(1,2,3))
    
    val sliced_3f = fSlicer.transform(lshifted)

    
    //Step8: put everything together in a pipeline
   var pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect , lSlicer,v2d , lShifter, fSlicer))

    
    //Step9: generate model by fitting the rawDf into the pipeline
   var pipelineModel = (pipeline: Pipeline).fit(obsDF)

    //Step10: transform data with the model - do predictions
   val tansformed =  pipelineModel.transform(obsDF)
  

    //Step11: drop all columns from the dataframe other than label and features
   val final_df = tansformed.select( "label", "features")
    
   final_df.collect().foreach(println)
    
    
    val myLR  = new LinearRegression().setMaxIter(3).setRegParam(0.001).setElasticNetParam(0.1)
  
  
 
    val lrStage = 0
    //val params = ParamMap(myLR.maxIter -> 10 , myLR.regParam -> 0.3)
    
     pipeline = new Pipeline().setStages(Array( myLR))
     pipelineModel = pipeline.fit(final_df)
    
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