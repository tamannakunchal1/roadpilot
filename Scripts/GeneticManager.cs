using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class GeneticManager : MonoBehaviour
{
    [Header("References")]
    public CarController controller;

    [Header("Controls")]
    public int initialPopulation = 85;

    [Range(0.0f, 1.0f)]
    public float mutationRate = 0.055f;

    [Header("Crossover Controls")]
    public int bestAgentSelection = 8;
    public int worstAgentSelection = 3;
    public int numberToCrossover = 39;

    [Header("Public View")]
    public int currentGeneration = 0;
    public int currentGenome = 0;

    private List<int> genePool = new List<int>();

    private int naturallySelected = 0;

    private NNet[] population;

    // Prevents Death from advancing the genome more than once
    private bool isChangingGenome = false;


    // ============================================================
    // START
    // ============================================================

    private void Start()
    {
        if (controller == null)
        {
            Debug.LogError(
                "GeneticManager: Controller is not assigned."
            );

            return;
        }

        if (initialPopulation < 2)
        {
            initialPopulation = 2;
        }

        if (bestAgentSelection < 1)
        {
            bestAgentSelection = 1;
        }

        if (bestAgentSelection > initialPopulation)
        {
            bestAgentSelection = initialPopulation;
        }

        if (worstAgentSelection < 0)
        {
            worstAgentSelection = 0;
        }

        if (worstAgentSelection > initialPopulation)
        {
            worstAgentSelection = initialPopulation;
        }

        if (numberToCrossover < 0)
        {
            numberToCrossover = 0;
        }

        CreatePopulation();
    }


    // ============================================================
    // CREATE POPULATION
    // ============================================================

    private void CreatePopulation()
    {
        population =
            new NNet[initialPopulation];

        FillPopulationWithRandomValues(
            population,
            0
        );

        currentGeneration = 0;
        currentGenome = 0;

        ResetToCurrentGenome();
    }


    // ============================================================
    // RESET TO CURRENT GENOME
    // ============================================================

    private void ResetToCurrentGenome()
    {
        if (population == null)
        {
            Debug.LogError(
                "GeneticManager: Population is null."
            );

            return;
        }

        if (
            currentGenome < 0 ||
            currentGenome >= population.Length
        )
        {
            Debug.LogError(
                "GeneticManager: Invalid genome index: "
                + currentGenome
            );

            return;
        }

        if (population[currentGenome] == null)
        {
            Debug.LogError(
                "GeneticManager: Current network is null."
            );

            return;
        }

        controller.ResetWithNetwork(
            population[currentGenome]
        );
    }


    // ============================================================
    // FILL POPULATION WITH RANDOM NETWORKS
    // ============================================================

    private void FillPopulationWithRandomValues(
        NNet[] newPopulation,
        int startingIndex
    )
    {
        if (newPopulation == null)
        {
            return;
        }

        while (
            startingIndex <
            newPopulation.Length
        )
        {
            newPopulation[startingIndex] =
                new NNet();

            newPopulation[startingIndex].Initialise(
                controller.LAYERS,
                controller.NEURONS
            );

            newPopulation[startingIndex].fitness = 0f;

            startingIndex++;
        }
    }


    // ============================================================
    // DEATH
    // ============================================================

    public void Death(
        float fitness,
        NNet network
    )
    {
        // Prevent multiple Death calls from processing
        // the same genome.
        if (isChangingGenome)
        {
            return;
        }

        isChangingGenome = true;


        // --------------------------------------------------------
        // Store fitness of current genome
        // --------------------------------------------------------

        if (
            population != null &&
            currentGenome >= 0 &&
            currentGenome < population.Length
        )
        {
            population[currentGenome].fitness =
                fitness;
        }


        // --------------------------------------------------------
        // Move to next genome
        // --------------------------------------------------------

        if (
            currentGenome <
            population.Length - 1
        )
        {
            currentGenome++;

            ResetToCurrentGenome();

            isChangingGenome = false;

            return;
        }


        // --------------------------------------------------------
        // All genomes have died
        // Create new generation
        // --------------------------------------------------------

        RePopulate();

        isChangingGenome = false;
    }


    // ============================================================
    // REP0PULATE
    // ============================================================

    private void RePopulate()
    {
        if (population == null)
        {
            return;
        }

        genePool.Clear();

        naturallySelected = 0;

        currentGeneration++;


        // --------------------------------------------------------
        // Sort from best fitness to worst fitness
        // --------------------------------------------------------

        SortPopulation();


        // --------------------------------------------------------
        // Select best networks
        // --------------------------------------------------------

        NNet[] newPopulation =
            PickBestPopulation();


        // --------------------------------------------------------
        // Crossover
        // --------------------------------------------------------

        Crossover(
            newPopulation
        );


        // --------------------------------------------------------
        // Mutate
        // --------------------------------------------------------

        Mutate(
            newPopulation
        );


        // --------------------------------------------------------
        // Fill remaining slots with random networks
        // --------------------------------------------------------

        FillPopulationWithRandomValues(
            newPopulation,
            naturallySelected
        );


        // --------------------------------------------------------
        // Replace population
        // --------------------------------------------------------

        population =
            newPopulation;


        // --------------------------------------------------------
        // Start from first genome
        // --------------------------------------------------------

        currentGenome = 0;

        ResetToCurrentGenome();


        Debug.Log(
            "Generation "
            + currentGeneration
            + " started."
        );
    }


    // ============================================================
    // MUTATION
    // ============================================================

    private void Mutate(
        NNet[] newPopulation
    )
    {
        if (newPopulation == null)
        {
            return;
        }

        for (
            int i = 0;
            i < naturallySelected &&
            i < newPopulation.Length;
            i++
        )
        {
            if (newPopulation[i] == null)
            {
                continue;
            }


            // ----------------------------------------------------
            // Mutate weights
            // ----------------------------------------------------

            for (
                int c = 0;
                c < newPopulation[i].weights.Count;
                c++
            )
            {
                if (
                    Random.Range(
                        0.0f,
                        1.0f
                    ) < mutationRate
                )
                {
                    newPopulation[i].weights[c] =
                        MutateMatrix(
                            newPopulation[i].weights[c]
                        );
                }
            }


            // ----------------------------------------------------
            // Mutate biases
            // ----------------------------------------------------

            for (
                int b = 0;
                b < newPopulation[i].biases.Count;
                b++
            )
            {
                if (
                    Random.Range(
                        0.0f,
                        1.0f
                    ) < mutationRate
                )
                {
                    newPopulation[i].biases[b] =
                        Mathf.Clamp(
                            newPopulation[i].biases[b]
                            +
                            Random.Range(
                                -0.5f,
                                0.5f
                            ),
                            -1f,
                            1f
                        );
                }
            }
        }
    }


    // ============================================================
    // MUTATE MATRIX
    // ============================================================

    private Matrix<float> MutateMatrix(
        Matrix<float> matrix
    )
    {
        if (matrix == null)
        {
            return matrix;
        }

        int totalValues =
            matrix.RowCount *
            matrix.ColumnCount;

        if (totalValues <= 0)
        {
            return matrix;
        }


        // The old code could create:
        //
        // Random.Range(1, 0)
        //
        // for small matrices.
        //
        // This prevents that problem.

        int maximumPoints =
            Mathf.Max(
                1,
                totalValues / 7
            );

        int randomPoints =
            Random.Range(
                1,
                maximumPoints + 1
            );


        // Clone instead of modifying the parent's matrix.
        Matrix<float> result =
            matrix.Clone();


        for (
            int i = 0;
            i < randomPoints;
            i++
        )
        {
            int randomColumn =
                Random.Range(
                    0,
                    result.ColumnCount
                );

            int randomRow =
                Random.Range(
                    0,
                    result.RowCount
                );


            result[randomRow, randomColumn] =
                Mathf.Clamp(
                    result[randomRow, randomColumn]
                    +
                    Random.Range(
                        -1f,
                        1f
                    ),
                    -1f,
                    1f
                );
        }

        return result;
    }


    // ============================================================
    // CROSSOVER
    // ============================================================

    private void Crossover(
        NNet[] newPopulation
    )
    {
        if (newPopulation == null)
        {
            return;
        }

        if (population == null)
        {
            return;
        }


        // IMPORTANT:
        //
        // numberToCrossover = 39
        //
        // The loop increases by 2:
        //
        // 0, 2, 4, ... 38
        //
        // Therefore it creates 20 pairs = 40 children.
        //
        // This matches the original project's behavior.


        for (
            int i = 0;
            i < numberToCrossover;
            i += 2
        )
        {
            // ----------------------------------------------------
            // Make sure there is room for two children
            // ----------------------------------------------------

            if (
                naturallySelected + 1 >=
                newPopulation.Length
            )
            {
                break;
            }


            int AIndex = 0;
            int BIndex = 0;


            // ----------------------------------------------------
            // Choose parents from gene pool
            // ----------------------------------------------------

            if (genePool.Count >= 1)
            {
                AIndex =
                    genePool[
                        Random.Range(
                            0,
                            genePool.Count
                        )
                    ];

                BIndex =
                    genePool[
                        Random.Range(
                            0,
                            genePool.Count
                        )
                    ];


                // Try to select different parents
                int attempts = 0;

                while (
                    AIndex == BIndex &&
                    attempts < 100 &&
                    genePool.Count > 1
                )
                {
                    BIndex =
                        genePool[
                            Random.Range(
                                0,
                                genePool.Count
                            )
                        ];

                    attempts++;
                }
            }
            else
            {
                // ------------------------------------------------
                // If gene pool is empty, select random parents
                // ------------------------------------------------

                AIndex =
                    Random.Range(
                        0,
                        population.Length
                    );

                BIndex =
                    Random.Range(
                        0,
                        population.Length
                    );


                int attempts = 0;

                while (
                    AIndex == BIndex &&
                    population.Length > 1 &&
                    attempts < 100
                )
                {
                    BIndex =
                        Random.Range(
                            0,
                            population.Length
                        );

                    attempts++;
                }
            }


            // ----------------------------------------------------
            // Validate parent indexes
            // ----------------------------------------------------

            if (
                AIndex < 0 ||
                AIndex >= population.Length ||
                BIndex < 0 ||
                BIndex >= population.Length
            )
            {
                continue;
            }


            if (
                population[AIndex] == null ||
                population[BIndex] == null
            )
            {
                continue;
            }


            // ----------------------------------------------------
            // Create children
            // ----------------------------------------------------

            NNet Child1 =
                new NNet();

            NNet Child2 =
                new NNet();


            Child1.Initialise(
                controller.LAYERS,
                controller.NEURONS
            );

            Child2.Initialise(
                controller.LAYERS,
                controller.NEURONS
            );


            Child1.fitness = 0f;
            Child2.fitness = 0f;


            // ----------------------------------------------------
            // Crossover weights
            // ----------------------------------------------------

            int weightCount =
                Mathf.Min(
                    Child1.weights.Count,
                    population[AIndex].weights.Count
                );

            weightCount =
                Mathf.Min(
                    weightCount,
                    population[BIndex].weights.Count
                );


            for (
                int w = 0;
                w < weightCount;
                w++
            )
            {
                if (
                    Random.Range(
                        0.0f,
                        1.0f
                    ) < 0.5f
                )
                {
                    Child1.weights[w] =
                        population[AIndex]
                            .weights[w]
                            .Clone();

                    Child2.weights[w] =
                        population[BIndex]
                            .weights[w]
                            .Clone();
                }
                else
                {
                    Child1.weights[w] =
                        population[BIndex]
                            .weights[w]
                            .Clone();

                    Child2.weights[w] =
                        population[AIndex]
                            .weights[w]
                            .Clone();
                }
            }


            // ----------------------------------------------------
            // Crossover biases
            // ----------------------------------------------------

            int biasCount =
                Mathf.Min(
                    Child1.biases.Count,
                    population[AIndex].biases.Count
                );

            biasCount =
                Mathf.Min(
                    biasCount,
                    population[BIndex].biases.Count
                );


            for (
                int w = 0;
                w < biasCount;
                w++
            )
            {
                if (
                    Random.Range(
                        0.0f,
                        1.0f
                    ) < 0.5f
                )
                {
                    Child1.biases[w] =
                        population[AIndex]
                            .biases[w];

                    Child2.biases[w] =
                        population[BIndex]
                            .biases[w];
                }
                else
                {
                    Child1.biases[w] =
                        population[BIndex]
                            .biases[w];

                    Child2.biases[w] =
                        population[AIndex]
                            .biases[w];
                }
            }


            // ----------------------------------------------------
            // Add children to new population
            // ----------------------------------------------------

            newPopulation[
                naturallySelected
            ] =
                Child1;

            naturallySelected++;


            newPopulation[
                naturallySelected
            ] =
                Child2;

            naturallySelected++;
        }
    }


    // ============================================================
    // PICK BEST POPULATION
    // ============================================================

    private NNet[] PickBestPopulation()
    {
        NNet[] newPopulation =
            new NNet[initialPopulation];


        naturallySelected = 0;


        // --------------------------------------------------------
        // Select best agents
        // --------------------------------------------------------

        int bestCount =
            Mathf.Min(
                bestAgentSelection,
                population.Length
            );


        for (
            int i = 0;
            i < bestCount;
            i++
        )
        {
            if (population[i] == null)
            {
                continue;
            }


            if (
                naturallySelected >=
                newPopulation.Length
            )
            {
                break;
            }


            // Keep a copy of the best network
            newPopulation[
                naturallySelected
            ] =
                population[i].InitialiseCopy(
                    controller.LAYERS,
                    controller.NEURONS
                );


            newPopulation[
                naturallySelected
            ].fitness = 0f;


            naturallySelected++;


            // ----------------------------------------------------
            // Add this parent to gene pool according to fitness
            // ----------------------------------------------------

            int f =
                Mathf.RoundToInt(
                    Mathf.Max(
                        0f,
                        population[i].fitness
                    ) * 10f
                );


            // Prevent an extremely large gene pool
            f =
                Mathf.Min(
                    f,
                    100
                );


            for (
                int c = 0;
                c < f;
                c++
            )
            {
                genePool.Add(i);
            }
        }


        // --------------------------------------------------------
        // Add worst agents to gene pool
        // --------------------------------------------------------

        int worstCount =
            Mathf.Min(
                worstAgentSelection,
                population.Length
            );


        for (
            int i = 0;
            i < worstCount;
            i++
        )
        {
            int last =
                population.Length -
                1 -
                i;


            if (
                last < 0 ||
                last >= population.Length
            )
            {
                continue;
            }


            if (
                population[last] == null
            )
            {
                continue;
            }


            int f =
                Mathf.RoundToInt(
                    Mathf.Max(
                        0f,
                        population[last].fitness
                    ) * 10f
                );


            f =
                Mathf.Min(
                    f,
                    50
                );


            for (
                int c = 0;
                c < f;
                c++
            )
            {
                genePool.Add(last);
            }
        }


        return newPopulation;
    }


    // ============================================================
    // SORT POPULATION
    // ============================================================

    private void SortPopulation()
    {
        if (population == null)
        {
            return;
        }


        for (
            int i = 0;
            i < population.Length - 1;
            i++
        )
        {
            for (
                int j = i + 1;
                j < population.Length;
                j++
            )
            {
                if (
                    population[i] == null
                )
                {
                    continue;
                }


                if (
                    population[j] == null
                )
                {
                    NNet tempNull =
                        population[i];

                    population[i] =
                        population[j];

                    population[j] =
                        tempNull;

                    continue;
                }


                if (
                    population[i].fitness
                    <
                    population[j].fitness
                )
                {
                    NNet temp =
                        population[i];

                    population[i] =
                        population[j];

                    population[j] =
                        temp;
                }
            }
        }
    }
}