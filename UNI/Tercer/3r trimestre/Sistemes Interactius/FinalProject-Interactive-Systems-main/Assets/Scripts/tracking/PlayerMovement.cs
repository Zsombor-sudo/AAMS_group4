using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PlayerMovement : MonoBehaviour
{
    // Start is called before the first frame update
    public Quaternion q;
    public bool manual;
    public bool is_free;

    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void setPosition(Vector3 pos)
    {
        //swith playerIndex
        transform.position = pos;
    }

    public void setRotation(Quaternion quat)
    {
        Matrix4x4 mat = Matrix4x4.Rotate(quat);
        transform.localRotation = quat;
    }
}
