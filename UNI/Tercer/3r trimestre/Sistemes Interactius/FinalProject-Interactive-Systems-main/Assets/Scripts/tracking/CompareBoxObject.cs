using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CompareBoxObject : MonoBehaviour
{

    public string Tag;
    public bool is_empty;

    private void OnTriggerEnter(Collider other)
    {
    
       Tag = other.gameObject.tag;

    }
}
