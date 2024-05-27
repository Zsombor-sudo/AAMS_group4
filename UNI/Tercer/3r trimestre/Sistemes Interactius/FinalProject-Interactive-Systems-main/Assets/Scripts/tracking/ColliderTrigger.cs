using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ColliderTrigger : MonoBehaviour
{
    public string tagplayer1;
    public string tagplayer2;
    public GameObject fossil; 
    
    public GameObject player1; 
    public GameObject player2;



    private void OnTriggerEnter(Collider other) 
    {
        if (other.CompareTag(tagplayer1) && player1.GetComponent<PlayerMovement>().is_free == true) 
        {
            fossil.GetComponent<ObjectsMovement>().istrigger1 = true;
            fossil.GetComponent<ObjectsMovement>().touch_plane = false;
            player1.GetComponent<PlayerMovement>().is_free = false;

        }
        if (other.CompareTag(tagplayer2) && player2.GetComponent<PlayerMovement>().is_free == true) 
        {
            fossil.GetComponent<ObjectsMovement>().istrigger2 = true; 
            player2.GetComponent<PlayerMovement>().is_free = false;
            fossil.GetComponent<ObjectsMovement>().touch_plane = false;

        }

        
       
    }
}
