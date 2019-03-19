using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;

public class TakePhotos2 : MonoBehaviour {
    public int capx;
    public int capy;
    public int capwidth;
    public int capheight;

    //string srcFileName = @"D:/ufiles/b.txt";
    //string destFileName = @"D:/ufiles/c.txt";

    private int i;
    private string lastName;
    private string modify;
	int takephoto=0;
	string dir = @"../control";
    string inform = "";
    string response = "";
    string extension = "";

    public static int value;

    //if (File.Exists(srcFileName))
    //{                

    //}            

    //if (Directory.Exists(srcFolderPath))       
    //{                
    //    Directory.Move(srcFolderPath, destFolderPath);            
    //}

    // Use this for initialization
    void Start () {
		takephoto = 0;
        System.Random ran = new System.Random();
        i = ran.Next(10000,99999);
        modify = Convert.ToString(i);

        //在程序刚启动的时候，应该把inform,return文件名获取一遍，存在对应的全局变量中
        
        inform = getfilename(dir, "inform");
        response = getfilename(dir, "response");
        //string extension = ".txt";

    }
	
	// Update is called once per frame
	public void Update () {
		//Debug.Log (takephoto);
		if (Input.GetKeyDown (KeyCode.A))
        {
			Debug.Log (takephoto);
            //自定义截屏 
            StartCoroutine (getScreenTexture());
            //unity 自带截屏，只能是截全屏 
            //Application.CaptureScreenshot("shot.png");

            //lastName = srcFileName;
            //System.IO.File.Move(srcFileName, destFileName);

            //当unity更新图片后,修改inform文件名
            System.Random ran = new System.Random();
            int n = ran.Next(10000,99999);
            string generate = Convert.ToString(n);//随机数转字符串
            string new_inform ="inform"+ generate;
            System.IO.File.Move(dir + @"/" + inform + extension, dir + @"/" + new_inform + @".txt");//改名
            inform = getfilename(dir,"inform");//自己下次就知道inform是啥了

            //改完inform文件之后,就循环等待python改文件了
            while (true)
            {
                string check = getfilename(dir, "response");//不断获取关键字值response的文件名
                if (response.CompareTo(check) != 0)
                {//如果文件名被更改，跳出等待
                    response = check;
                    break;
                }
            }
            string value_str = (getfilename(dir, "value")).Substring(5, 1);//获取返回值
            value = Int32.Parse(value_str);//转出整数
            //Debug.Log(value);
        }
    }

    IEnumerator getScreenTexture()
    {
        yield return new WaitForEndOfFrame();
        //需要正确设置好图片保存格式 
        Texture2D t = new Texture2D(capwidth, capheight, TextureFormat.RGB24, true);
        //按照设定区域读取像素；注意是以左下角为原点读取 
        t.ReadPixels(new Rect(capx, capy, capwidth, capheight), 0, 0, false);
        t.Apply();
        //二进制转换 
        byte[] byt = t.EncodeToPNG();
        System.IO.File.WriteAllBytes(Application.streamingAssetsPath + "/111.png", byt);
    }

    string getfilename(string dir, string keyword)
    {
        DirectoryInfo folder = new DirectoryInfo(dir);//获取该路径下所有文件信息

        foreach (FileInfo file in folder.GetFiles("*.*")) //遍历所有文件
        {
            string filename = System.IO.Path.GetFileName(file.FullName);//获取除去路径的文件名(不含后缀)
            if (filename.Length >= keyword.Length)
            {//判断文件名长度是不是至少可关键字一样
                string key = filename.Substring(0, keyword.Length);//截取关键字长度的字符
                if (keyword.CompareTo(key) == 0)
                {//判断是不是和关键字一样
                    return filename;
                }
            }
        }
        return "";
    }

}
