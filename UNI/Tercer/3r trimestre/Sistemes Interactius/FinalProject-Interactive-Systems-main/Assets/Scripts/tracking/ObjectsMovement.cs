using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectsMovement : MonoBehaviour
{
    public GameObject player1; 
    public GameObject player2; 
    public GameObject box1; 
    public GameObject box2; 

    public string tagbox1;
    public string tagbox2;

    public bool istrigger1;
    public bool istrigger2;  

    public float DestroyDelay; 

    public string collidertag1;
    public string collidertag2;

    public string tagplane;
    Vector3 originalPos;
    public bool touch_plane;

    public bool box1_empty;
    public bool box2_empty;

    // Start is called before the first frame update
    void Start()
    {
        originalPos = gameObject.transform.position;
        box1_empty = box1.GetComponent<CompareBoxObject>().is_empty;
        box2_empty = box2.GetComponent<CompareBoxObject>().is_empty;
    }

    // Update is called once per frame
    void Update()
    {
        if (istrigger1){
            
            transform.position = player1.transform.position;
            if(touch_plane){
                transform.position = originalPos;
                touch_plane = false;
                istrigger1 = false;
                player1.GetComponent<PlayerMovement>().is_free = true;

            }
            
        }
        
        if (istrigger2){
            transform.position = player2.transform.position;
             if(touch_plane){
                transform.position = originalPos;
                touch_plane = false;
                istrigger2 = false;
                player2.GetComponent<PlayerMovement>().is_free = true;

            }
    
        }
    }

    private void OnTriggerEnter(Collider other) 
    {
         
        if (other.CompareTag(tagbox1)) 
        {  
            box1_empty = true;
            print(box1_empty);
            print(box2_empty);

            if(box1_empty && box2_empty){

                collidertag2 = box2.GetComponent<CompareBoxObject>().Tag;
                if(gameObject.tag == collidertag2){

                    istrigger1 = false; 
                    transform.position = box1.transform.position;
                    player1.GetComponent<PlayerMovement>().is_free = true;
                    Destroy(gameObject, DestroyDelay);
                    box1_empty = false;
                
                }else{
                    transform.position = originalPos;
                    box1_empty = false;
                }

            }
            
        }

        if (other.CompareTag(tagbox2)) 
        {
            box2_empty = true;
            print(box1_empty);
            print(box2_empty);
            if(box1_empty&& box2_empty){

                collidertag1 = box1.GetComponent<CompareBoxObject>().Tag;

                if(gameObject.tag == collidertag1){
                    istrigger2 = false; 
                    transform.position = box2.transform.position;
                    player2.GetComponent<PlayerMovement>().is_free = true;
                    Destroy(gameObject, DestroyDelay);
                    box2_empty = false;

                } else{
                    transform.position = originalPos;
                    box2_empty = false;
                }
                
            }
            
            
        }

        if(other.CompareTag(tagplane)){
            touch_plane = true;

        }

    }

    


}
