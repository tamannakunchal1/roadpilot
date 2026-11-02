using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    private Vector3 startPosition;
    private Vector3 startRotation;

    private NNet network;

    [Range(-1f, 1f)]
    public float a, t;

    public float timeSinceStart = 0f;

    // Prevent Death() from being called multiple times
    private bool isDead = false;

    [Header("Fitness")]
    public float overallFitness;
    public float distanceMultipler = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 0.1f;

    [Header("Network Options")]
    public int LAYERS = 1;
    public int NEURONS = 10;

    private Vector3 lastPosition;
    private float totalDistanceTravelled;
    private float avgSpeed;

    private float aSensor;
    private float bSensor;
    private float cSensor;


    // ============================================================
    // AWAKE
    // ============================================================

    private void Awake()
    {
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
    }


    // ============================================================
    // RESET WITH NETWORK
    // ============================================================

    public void ResetWithNetwork(NNet net)
    {
        network = net;

        Reset();
    }


    // ============================================================
    // RESET CAR
    // ============================================================

    public void Reset()
    {
        timeSinceStart = 0f;

        totalDistanceTravelled = 0f;

        avgSpeed = 0f;

        overallFitness = 0f;

        lastPosition = startPosition;

        // Allow the new genome to run
        isDead = false;

        transform.position = startPosition;

        transform.eulerAngles = startRotation;

        // Reset outputs
        a = 0f;
        t = 0f;

        // Reset sensors
        aSensor = 0f;
        bSensor = 0f;
        cSensor = 0f;
    }


    // ============================================================
    // COLLISION
    // ============================================================

    private void OnCollisionEnter(Collision collision)
    {
        if (isDead)
            return;

        if (collision.gameObject.CompareTag("Wall"))
        {
            Death();
        }
    }

    // ============================================================
    // FIXED UPDATE
    // ============================================================

    private void FixedUpdate()
    {
        // Don't continue driving after death
        if (isDead)
            return;

        InputSensors();

        lastPosition = transform.position;


        // Make sure GeneticManager has assigned a network
        if (network == null)
        {
            Debug.LogError(
                "CarController: Neural network has not been assigned."
            );

            Death();

            return;
        }


        // Run neural network
        (a, t) = network.RunNetwork(
            aSensor,
            bSensor,
            cSensor
        );


        // Move car
        MoveCar(a, t);


        // Update timer
        timeSinceStart += Time.fixedDeltaTime;


        // Calculate fitness
        CalculateFitness();
    }


    // ============================================================
    // DEATH
    // ============================================================

    private void Death()
    {
        // Prevent multiple Death() calls
        if (isDead)
            return;

        isDead = true;


        // Save fitness to current network
        if (network != null)
        {
            network.fitness = overallFitness;
        }


        // Find GeneticManager
        GeneticManager manager =
            FindAnyObjectByType<GeneticManager>();


        if (manager != null)
        {
            manager.Death(
                overallFitness,
                network
            );
        }
        else
        {
            Debug.LogError(
                "CarController: GeneticManager was not found in the scene!"
            );
        }
    }


    // ============================================================
    // FITNESS
    // ============================================================

    private void CalculateFitness()
    {
        if (isDead)
            return;


        totalDistanceTravelled +=
            Vector3.Distance(
                transform.position,
                lastPosition
            );


        // Prevent division by zero
        if (timeSinceStart > 0f)
        {
            avgSpeed =
                totalDistanceTravelled /
                timeSinceStart;
        }
        else
        {
            avgSpeed = 0f;
        }


        overallFitness =
            (totalDistanceTravelled *
             distanceMultipler)

            +

            (avgSpeed *
             avgSpeedMultiplier)

            +

            (
                (
                    (aSensor + bSensor + cSensor)
                    / 3f
                )
                *
                sensorMultiplier
            );


        // Kill car if it has been running
        // for too long without making progress
        if (
            timeSinceStart > 20f &&
            overallFitness < 40f
        )
        {
            Death();

            return;
        }


        // Optional maximum fitness condition
        if (overallFitness >= 1000f)
        {
            Death();

            return;
        }
    }


    // ============================================================
    // SENSORS
    // ============================================================

    private void InputSensors()
    {
        if (isDead)
            return;


        // Right sensor
        Vector3 a =
            transform.forward +
            transform.right;


        // Center sensor
        Vector3 b =
            transform.forward;


        // Left sensor
        Vector3 c =
            transform.forward -
            transform.right;


        Ray r =
            new Ray(
                transform.position,
                a
            );

        RaycastHit hit;


        // --------------------------------------------------------
        // RIGHT SENSOR
        // --------------------------------------------------------

        if (
            Physics.Raycast(
                r,
                out hit
            )
        )
        {
            aSensor =
                hit.distance / 20f;

            Debug.DrawLine(
                r.origin,
                hit.point,
                Color.red
            );
        }
        else
        {
            aSensor = 1f;
        }


        // --------------------------------------------------------
        // CENTER SENSOR
        // --------------------------------------------------------

        r.direction = b;

        if (
            Physics.Raycast(
                r,
                out hit
            )
        )
        {
            bSensor =
                hit.distance / 20f;

            Debug.DrawLine(
                r.origin,
                hit.point,
                Color.red
            );
        }
        else
        {
            bSensor = 1f;
        }


        // --------------------------------------------------------
        // LEFT SENSOR
        // --------------------------------------------------------

        r.direction = c;

        if (
            Physics.Raycast(
                r,
                out hit
            )
        )
        {
            cSensor =
                hit.distance / 20f;

            Debug.DrawLine(
                r.origin,
                hit.point,
                Color.red
            );
        }
        else
        {
            cSensor = 1f;
        }
    }


    // ============================================================
    // MOVE CAR
    // ============================================================

    private Vector3 inp;

    public void MoveCar(
        float v,
        float h
    )
    {
        if (isDead)
            return;


        inp =
            Vector3.Lerp(
                Vector3.zero,
                new Vector3(
                    0,
                    0,
                    v * 11.4f
                ),
                0.02f
            );


        inp =
            transform.TransformDirection(
                inp
            );


        transform.position += inp;


        transform.eulerAngles +=
            new Vector3(
                0,
                (h * 90f) * 0.02f,
                0
            );
    }
}