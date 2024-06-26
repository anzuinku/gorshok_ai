package com.example.gorshok_ai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.example.gorshok_ai.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.text.DecimalFormat

class MainActivity : AppCompatActivity() {

    lateinit var selectBtn: Button
    lateinit var predBtn: Button
    lateinit var takeBtn: Button
    lateinit var res1View: TextView
    lateinit var res2View: TextView
    lateinit var res3View: TextView
    lateinit var res4View: TextView
    lateinit var res5View: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        getPermission()

        selectBtn=findViewById(R.id.selectBtn)
        takeBtn=findViewById(R.id.takeBtn)
        predBtn=findViewById(R.id.predictBtn)
        res1View=findViewById(R.id.res1View)
        res2View=findViewById(R.id.res2View)
        res3View=findViewById(R.id.res3View)
        res4View=findViewById(R.id.res4View)
        res5View=findViewById(R.id.res5View)
        imageView=findViewById(R.id.imageView)

        val labels=application.assets.open("labels.txt").bufferedReader().readLines()

        selectBtn.setOnClickListener{
            val intent= Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent,100)
        }
        takeBtn.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, 200)
        }
        predBtn.setOnClickListener{
            val cropSize = Math.min(bitmap.width, bitmap.height)
            //image processor
            val imageProcessor= ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0.0f,255.0f))
                .build()

            var tensorImage= TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            tensorImage= imageProcessor .process(tensorImage)

            val model = Model.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            var res =""
            //var maxIdx=0
            val mform = DecimalFormat("#.##")
            outputFeature0.floatArray.forEachIndexed{index,fl->
                if (index<5){
                    if (index==0) {
                        res += labels[index] + ": " + mform.format(outputFeature0.floatArray[index] * 100) + "%"
                        res1View.setText(res)
                    }
                    else if (index==1) {
                        res += labels[index] + ": " + mform.format(outputFeature0.floatArray[index] * 100) + "%"
                        res2View.setText(res)
                    }
                    else if (index==2) {
                        res += labels[index] + ": " + mform.format(outputFeature0.floatArray[index] * 100) + "%"
                        res3View.setText(res)
                    }
                    else if (index==3) {
                        res += labels[index] + ": " + mform.format(outputFeature0.floatArray[index] * 100) + "%"
                        res4View.setText(res)
                    }
                    else if (index==4) {
                        res += labels[index] + ": " + mform.format(outputFeature0.floatArray[index] * 100) + "%"
                        res5View.setText(res)
                    }
            }
            res=""}
            // Releases model resources if no longer used.
            model.close()}
    }

    fun getPermission() {
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 11)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode==11){
            if (grantResults.size>0){
                if (grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                    this.getPermission()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?)
    {
        super.onActivityResult(requestCode,resultCode,data)
        if(requestCode==100){
            val uri=data?.data
            bitmap= MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            imageView.setImageBitmap(bitmap)
        }
        else if(requestCode==200){
            bitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(bitmap)
        }
    }
}
