import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS


object ALSPractice {

  case class User(userId:String, name:String, sex:String, _userId:Int)
  case class Order(userId:String, orderId:String, date:String)
  case class OrderItem(orderId:String, itemId:String, _itemId:Int)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()

    val userData = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferSchema", true)
      .schema(ScalaReflection.schemaFor[User].dataType.asInstanceOf[StructType])
      .load("C:\\Users\\mirac\\Desktop\\课程资料\\chapter_5\\als_user.csv")

    val orderData = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferSchema", true)
      .schema(ScalaReflection.schemaFor[Order].dataType.asInstanceOf[StructType])
      .load("C:\\Users\\mirac\\Desktop\\课程资料\\chapter_5\\als_order.csv")

    val itemData = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferSchema", true)
      .schema(ScalaReflection.schemaFor[OrderItem].dataType.asInstanceOf[StructType])
      .load("C:\\Users\\mirac\\Desktop\\课程资料\\chapter_5\\als_item.csv")

//    userData.show(false)
//    orderData.show(false)
//    itemData.show(false)

    val _userItemData = userData.join(
      orderData,
      userData("userId") === orderData("userId")
    ).join(
      itemData,
      itemData("orderId") === orderData("orderId")
    ).select(
      userData("userId"),
      itemData("itemId"),
      userData("_userId"),
      itemData("_itemId")
    )

//    _userItemData.show(false)
    val _userItemRating = _userItemData.withColumn(
      "rating",
      col("_userId") * 0 + 1.0
    )

//    _userItemRating.show(false)

    val userItemRating = _userItemRating.groupBy(
      col("_userId"),
      col("_itemId")
    ).agg(sum("rating"))
      .withColumnRenamed("sum(rating)", "rating")

//    userItemRating.show(false)
    //划分数据集为训练集和测试集
    val Array(training, test) = userItemRating.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setItemCol("_itemId")
      .setUserCol("_userId")
      .setRatingCol("rating")
      .setMaxIter(5)
      .setRegParam(0.01)

    //训练模型
    val model = als.fit(training)
    //预测
    val prediction = model.transform(test)

    test.show(false)
    prediction.show(false)

    val regressionEvaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = regressionEvaluator.evaluate(prediction)
    println(rmse)
  }
}
