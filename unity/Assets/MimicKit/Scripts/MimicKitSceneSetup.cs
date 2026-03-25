using UnityEngine;

/// <summary>
/// Creates a basic scene with ground plane, camera, and lighting.
/// Attach to an empty GameObject or use as an Editor script.
/// </summary>
[ExecuteInEditMode]
public class MimicKitSceneSetup : MonoBehaviour
{
    [Tooltip("Run setup once on Start")]
    public bool setupOnStart = true;

    void Start()
    {
        if (setupOnStart && Application.isPlaying)
            SetupScene();
    }

    [ContextMenu("Setup Scene")]
    public void SetupScene()
    {
        // Ground plane
        if (GameObject.Find("Ground") == null)
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.position = Vector3.zero;
            ground.transform.localScale = new Vector3(10, 1, 10);
            var mat = new Material(Shader.Find("Standard"));
            mat.color = new Color(0.16f, 0.16f, 0.29f);
            ground.GetComponent<Renderer>().material = mat;

            // Set ground to Default layer (collides with Humanoid layer 8)
            ground.layer = 0;
        }

        // Camera
        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(3, 2, 4);
            cam.transform.LookAt(new Vector3(0, 1, 0));
            cam.backgroundColor = new Color(0.1f, 0.1f, 0.18f);
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.farClipPlane = 100;
        }

        // Directional light
        var lights = FindObjectsByType<Light>(FindObjectsSortMode.None);
        bool hasDir = false;
        foreach (var l in lights)
            if (l.type == LightType.Directional) hasDir = true;

        if (!hasDir)
        {
            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            light.shadows = LightShadows.Soft;
            lightGo.transform.rotation = Quaternion.Euler(50, -30, 0);
        }
    }
}
