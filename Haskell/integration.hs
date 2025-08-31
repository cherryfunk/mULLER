-- Different approaches to integration in Haskell
integrationComparison :: IO ()
integrationComparison = do
  putStrLn "\n=== Integration Methods Comparison ==="

  -- Method 1: Using monad-bayes Integrator (what we showed earlier)
  putStrLn "\n1. Monad-Bayes Integrator (more points?):"
  let points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  let betterUniform = integrator (\f -> sum (map f points) / fromIntegral (length points))
  let integral1b = runIntegrator (\x -> x*x) betterUniform
  putStrLn $ "   Using " ++ show (length points) ++ " points: ∫ x² dx from 0 to 1 ≈ " ++ show integral1b

  -- Method 2: Trapezoidal rule via numeric-tools
  putStrLn "\n2. Trapezoidal Rule (numeric-tools):"
  let resTrap = quadTrapezoid defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resTrap)
           ++ ", prec≈ " ++ show (quadPrecEst resTrap)
           ++ ", iters= " ++ show (quadNIter resTrap)

  -- Method 3: Simpson's rule via numeric-tools
  putStrLn "\n3. Simpson's Rule (numeric-tools):"
  let resSimp = quadSimpson defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resSimp)
           ++ ", prec≈ " ++ show (quadPrecEst resSimp)
           ++ ", iters= " ++ show (quadNIter resSimp)

  -- Method 3b: Romberg via numeric-tools
  putStrLn "\n3b. Romberg (numeric-tools):"
  let resRomb = quadRomberg defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resRomb)
           ++ ", prec≈ " ++ show (quadPrecEst resRomb)
           ++ ", iters= " ++ show (quadNIter resRomb)

  -- Method 4: Analytical (when possible)
  putStrLn "\n4. Analytical Solution (Exact):"
  let analytical = 1/3 :: Double
  putStrLn $ "   ∫ x² dx from 0 to 1 = " ++ show analytical

  -- Method 5: Monte Carlo (simple implementation)
  putStrLn "\n5. Monte Carlo Integration:"
  let monteCarlo :: (Double -> Double) -> Double -> Double -> Int -> IO Double
      monteCarlo f a b n = do
        points <- sequence (replicate n (randomRIO (a, b)))
        let values = map f points
        return $ ((b - a) / fromIntegral n) * sum values
  integral5 <- monteCarlo (\x -> x*x) 0.0 1.0 10000
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show integral5

  -- Method 6: GaussQuadIntegration (professional numerical library)
  putStrLn "\n6. GaussQuadIntegration (Professional Library):"
  let integral6 = nIntegrate256 (\x -> x*x) 0.0 1.0
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show integral6

  -- Demonstrate different integration domains
  putStrLn "\n=== Integration Domain Capabilities ==="

  putStrLn "\nGaussQuadIntegration Capabilities:"
  putStrLn "✅ 1D intervals: [a,b]"
  putStrLn "✅ High precision (up to 1024 points)"
  putStrLn "✅ Arbitrary precision types (Fixed, etc.)"
  putStrLn "❌ Multi-dimensional domains"
  putStrLn "❌ Non-rectangular regions"
  putStrLn "❌ Complex geometries"

  -- Example: Different 1D intervals
  putStrLn $ "\nDifferent 1D examples with GaussQuadIntegration:"
  let int1 = nIntegrate256 (\x -> sin x) 0.0 pi
  putStrLn $ "∫ sin(x) dx from 0 to π = " ++ show int1 ++ " (exact: 2.0)"

  let int2 = nIntegrate256 (\x -> exp (-x*x)) (-1.0) 1.0
  putStrLn $ "∫ e^(-x²) dx from -1 to 1 ≈ " ++ show int2

  let int3 = nIntegrate256 (\x -> 1/sqrt(2*pi) * exp(-x*x/2)) (-3.0) 3.0
  putStrLn $ "∫ φ(x) dx from -3 to 3 (normal CDF) ≈ " ++ show int3 ++ " (should be ~1.0)"

  putStrLn "\n=== How Monad-Bayes Integrator Works ==="
  putStrLn "• integrator :: ((a -> Double) -> Double) -> Integrator a"
  putStrLn "• runIntegrator :: (a -> Double) -> Integrator a -> Double"
  putStrLn "• The 'integrator' function creates a measure from an integration kernel"
  putStrLn "• The kernel (a->Double)->Double defines how to integrate any function"
  putStrLn "• Our example: uniform on [0,1] using discrete point evaluation"

  putStrLn "\n=== Alternatives for Complex Domains ==="
  putStrLn "For multi-dimensional and complex geometries:"
  putStrLn "• adaptive-cubature: Multi-dimensional adaptive integration"
  putStrLn "• scubature: Integration over simplices (triangles/tetrahedrons)"
  putStrLn "• Monte Carlo methods: For very high-dimensional problems"
  putStrLn "• Domain decomposition: Break complex regions into simple pieces"

  putStrLn "\nExample complex domain approaches:"
  putStrLn "• 2D circle: Use polar coordinates + 1D integration"
  putStrLn "• 3D sphere: Use spherical coordinates + 2D integration"
  putStrLn "• Irregular regions: Use adaptive quadrature or triangulation"
  putStrLn "• Complex boundaries: Use Green's theorem or change of variables"

  putStrLn "\n=== Summary ==="
  putStrLn "• For exact results: Use analytical methods when possible"
  putStrLn "• For 1D intervals: GaussQuadIntegration ⭐ (fast, accurate)"
  putStrLn "• For multi-D: adaptive-cubature or scubature"
  putStrLn "• For complex geometries: Monte Carlo or domain decomposition"
  putStrLn "• For probabilistic programming: Use monad-bayes Integrator"
  putStrLn "• For general scientific computing: Use hmatrix ecosystem"
  putStrLn "• For simple cases: Implement trapezoidal/Simpson's rule directly"
  putStrLn "• For custom measures: Build your own integration kernel with Integrator"

-- Testing infinite lists in NeSy. Is it not supposed to handle infinite lists?
testInfiniteLists :: IO ()
testInfiniteLists = do
  putStrLn "\n=== Testing Infinite Lists in NeSy ==="

  -- Create an infinite list
  let infiniteDomain = [0.0, 0.1 ..] :: [Double]  -- Infinite: [0.0, 0.1, 0.2, ...]
  putStrLn $ "Infinite domain created: [0.0, 0.1, 0.2, ...] (lazy)"

  -- Test with finite prefix (safe)
  let finitePrefix = take 5 infiniteDomain
  putStrLn $ "Finite prefix: " ++ show finitePrefix

  -- Test quantifiers with finite prefix
  let testPred x = x > 0.5
  putStrLn $ "∃x ∈ finitePrefix: x > 0.5 = " ++ show (any testPred finitePrefix)
  putStrLn $ "∀x ∈ finitePrefix: x > 0.5 = " ++ show (all testPred finitePrefix)

  -- Now try with larger finite prefix
  let largerPrefix = take 20 infiniteDomain
  putStrLn $ "\nWith 20 elements:"
  putStrLn $ "∃x ∈ largerPrefix: x > 0.5 = " ++ show (any testPred largerPrefix)
  putStrLn $ "∀x ∈ largerPrefix: x > 0.5 = " ++ show (all testPred largerPrefix)

  -- Show what happens with very large prefix (approaches infinite behavior)
  let bigPrefix = take 100 infiniteDomain
  putStrLn $ "\nWith 100 elements (approximating infinite):"
  putStrLn $ "∃x ∈ bigPrefix: x > 0.5 = " ++ show (any testPred bigPrefix)
  putStrLn $ "∀x ∈ bigPrefix: x > 0.5 = " ++ show (all testPred bigPrefix)

  -- Demonstrate the problem: what happens if we try to use infinite directly?
  putStrLn $ "\n=== The Problem with True Infinite Lists ==="
  putStrLn "If we tried: foldr disj bot $ map testPred infiniteDomain"
  putStrLn "This would: 1) Never terminate, 2) Use infinite memory, 3) Loop forever"
  putStrLn "Because foldr would keep evaluating elements forever!"

  -- Show lazy evaluation to the rescue
  putStrLn $ "\n=== Lazy Evaluation Solution ==="
  let lazyExists pred list = any pred (take 1000 list)  -- Limit to 1000 elements
  let lazyForall pred list = all pred (take 1000 list)  -- Limit to 1000 elements

  putStrLn $ "Lazy ∃x: x > 0.5 (first 1000 elements) = " ++ show (lazyExists testPred infiniteDomain)
  putStrLn $ "Lazy ∀x: x > 0.5 (first 1000 elements) = " ++ show (lazyForall testPred infiniteDomain)