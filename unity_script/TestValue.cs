using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestValue : MonoBehaviour {
    public GameObject pig;
    public GameObject bird;
    public GameObject bear;
    public GameObject elephant;
    public GameObject seal;
    public GameObject monkey;
    public GameObject tiger;

    // Use this for initialization
    void Start () {
        TakePhotos2.value = 9;
        pig.SetActive(false);
        bird.SetActive(false);
        bear.SetActive(false);
        elephant.SetActive(false);
        seal.SetActive(false);
        monkey.SetActive(false);
        tiger.SetActive(false);
    }
	
	// Update is called once per frame
	void Update () {
        Appearance();
    }

    public void Appearance()
    {
        Debug.Log(TakePhotos2.value);
        if (TakePhotos2.value == 1)
        {
            pig.SetActive(true);
            bird.SetActive(false);
            bear.SetActive(false);
            elephant.SetActive(false);
            seal.SetActive(false);
            monkey.SetActive(false);
            tiger.SetActive(false);
            //SSDebug.Log(TakePhotos2.value);
        }

        if (TakePhotos2.value == 2)
        {
            pig.SetActive(false);
            bird.SetActive(true);
            bear.SetActive(false);
            elephant.SetActive(false);
            seal.SetActive(false);
            monkey.SetActive(false);
            tiger.SetActive(false);
        }

        if (TakePhotos2.value == 3)
        {
            pig.SetActive(false);
            bird.SetActive(false);
            bear.SetActive(true);
            elephant.SetActive(false);
            seal.SetActive(false);
            monkey.SetActive(false);
            tiger.SetActive(false);
        }

        if (TakePhotos2.value == 4)
        {
            pig.SetActive(false);
            bird.SetActive(false);
            bear.SetActive(false);
            elephant.SetActive(true);
            seal.SetActive(false);
            monkey.SetActive(false);
            tiger.SetActive(false);
        }

        if (TakePhotos2.value == 5)
        {
            pig.SetActive(false);
            bird.SetActive(false);
            bear.SetActive(false);
            elephant.SetActive(false);
            seal.SetActive(true);
            monkey.SetActive(false);
            tiger.SetActive(false);
        }

        if (TakePhotos2.value == 6)
        {
            pig.SetActive(false);
            bird.SetActive(false);
            bear.SetActive(false);
            elephant.SetActive(false);
            seal.SetActive(false);
            monkey.SetActive(true);
            tiger.SetActive(false);
        }

        if (TakePhotos2.value == 7)
        {
            pig.SetActive(false);
            bird.SetActive(false);
            bear.SetActive(false);
            elephant.SetActive(false);
            seal.SetActive(false);
            monkey.SetActive(false);
            tiger.SetActive(true);
        }
    }
}
