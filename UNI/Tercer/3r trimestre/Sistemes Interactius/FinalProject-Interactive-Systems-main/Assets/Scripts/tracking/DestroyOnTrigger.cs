using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DestroyOnTrigger : MonoBehaviour
{
    
    public string tagPlayer1;
    public string tagPlayer2;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag(tagPlayer1) || other.CompareTag(tagPlayer2))
        {
            Destroy(gameObject);
        }
    }

}
