using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.IO;
using UnityEngine;



public class PythonTest : MonoBehaviour {
    //public GameObject camera;

    // Use this for initialization
    void Start () {

        //@"F:\blockly-du\test_for_BAT\Du.BAT"
        //打开BAT
        string line = @"python C:\Users\63292\Desktop\MLAR\test01.py";
        //cmd(line);
        string ll = cmd(line);
        int n = 0; 
        n = ll.Length;
        //Console.WriteLine(ll);
        UnityEngine.Debug.Log(n);
        //Console.WriteLine(ll);

    }
	
	// Update is called once per frame
	void Update () {
        
    }

    static string cmd(string sr)
    {
        Process pro = null;

        string ll = string.Empty;
        try
        {
            pro = new Process();
            pro.StartInfo.FileName = "powershell";             //cmd
            pro.StartInfo.UseShellExecute = false;          //不显示shell
            pro.StartInfo.CreateNoWindow = false;            //不创建窗口
            pro.StartInfo.RedirectStandardInput = true;     //打开流输入
            pro.StartInfo.RedirectStandardOutput = true;    //打开流输出
            pro.StartInfo.RedirectStandardError = true;     //打开错误流

            pro.Start();//执行

            pro.StandardInput.WriteLine(sr);      //&exit运行完立即退出
            pro.StandardInput.WriteLine("exit");
            pro.StandardInput.AutoFlush = false;             //清缓存

            ll = pro.StandardOutput.ReadToEnd();            //读取输出


            pro.WaitForExit();                              //等待程序执行完退出进程  

            pro.Close();//结束  

            return ll;
        }
        catch (Exception ex)
        {
            Console.WriteLine("Exception Occurred :{0},{1}", ex.Message, ex.StackTrace.ToString());
            return null;
        }
    }

}
