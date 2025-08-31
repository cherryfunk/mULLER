{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}

module Main where

import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Maybe
import Data.List (foldl')
import Numeric.Tools.Integration (defQuad, quadTrapezoid, quadSimpson, quadRomberg, quadRes)

-- Algebra of truth values
class TwoMonBLat a where
  top :: a
  bot :: a
  neg :: a -> a
  conj :: a -> a -> a
  disj :: a -> a -> a
  implies :: a -> a -> a
  -- Default implementations
  neg a = a `implies` bot
  a `implies` b = neg a `disj` b

-- Simple Boolean instance
instance TwoMonBLat Bool where
  top = True
  bot = False
  neg = not
  conj = (&&)
  disj = (||)
  implies a b = not a || b

-- Fuzzy truth values in [0,1]
instance TwoMonBLat Double where
  top = 1.0
  bot = 0.0
  neg x = 1.0 - x
  conj a b = a * b
  disj a b = a + b - a * b
  implies a b = if a <= b then 1.0 else b / a
  

-- Domain representation
data Domain a where
  Finite :: [a] -> Domain a                            -- Finite list of elements
  CountableInfinite :: [a] -> Domain a                 -- Lazy infinite list
  RealInterval :: Double -> Double -> Domain Double    -- Real interval (a,b)

deriving instance Show a => Show (Domain a)

-- Aggregation over domains (b is any domain element type)
class (TwoMonBLat a) => Aggr2MonBLat a where
  aggrE, aggrA :: Domain b -> (b -> a) -> a


-- Quantifier aggregation for fuzzy truth with uniform density f(a)=1
instance Aggr2MonBLat Double where
  -- ∃ as ¬∀¬
  aggrE d p = 1.0 - aggrA d (\x -> 1.0 - p x)
  -- ∀ via geometric mean of truth values (q=1) with uniform density
  --  Integration using quadTrapezoid, however one could use other integration methods
  aggrA (Finite xs) p =
    case length xs of
      0 -> 1.0
      n -> exp (sum (map (log . p) xs) / fromIntegral n)
  aggrA (CountableInfinite xs) p =
    let prefix = take 1000 xs
    in if null prefix then 1.0
       else exp (sum (map (log . p) prefix) / fromIntegral (length prefix))
  aggrA (RealInterval a b) p =
    if a == b then 1.0
    else
      let integrand x = log (p x)
          res = quadTrapezoid defQuad (a, b) integrand
      in case quadRes res of
           Just val -> exp (val / (b - a))
           Nothing  -> 0.0/0.0

-- Syntax
type Ident = String

-- Terms
data Term =
    Var Ident
  | Const Double
  | Appl Ident [Term]
  deriving (Show)

-- Formulas
data Formula =
    T | F
  | Pred Ident [Term]
  | Not Formula
  | And Formula Formula
  | Or Formula Formula
  | Implies Formula Formula
  | Forall Ident Formula
  | Exists Ident Formula
  deriving (Show)

-- Interpretation specialized to Double
data Interpretation = Interpretation
  { domain     :: Domain Double
  , functions  :: Map.Map Ident ([Double] -> Double)
  , predicates :: Map.Map Ident ([Double] -> Double)
  }

-- Valuation (variable assignment)
type Valuation = Map.Map Ident Double

-- Evaluate terms
evalTerm :: Interpretation -> Valuation -> Term -> Double
evalTerm _     val (Var var) =
  case Map.lookup var val of
    Just v  -> v
    Nothing -> error ("Variable " ++ var ++ " not found")
evalTerm _     _   (Const c) = c
evalTerm interp val (Appl f args) =
  let argVals = map (evalTerm interp val) args
      func = case Map.lookup f (functions interp) of
               Just fn -> fn
               Nothing -> error ("Function " ++ f ++ " not found")
  in func argVals

-- Evaluate formulas
evalFormula :: Interpretation -> Valuation -> Formula -> Double
evalFormula _ _ T = 1.0
evalFormula _ _ F = 0.0
evalFormula interp val (Pred p args) =
  let argVals = map (evalTerm interp val) args
      predFn = case Map.lookup p (predicates interp) of
                 Just pr -> pr
                 Nothing -> error ("Predicate " ++ p ++ " not found")
  in predFn argVals
evalFormula interp val (Not f) = neg (evalFormula interp val f)
evalFormula interp val (And f1 f2) = conj (evalFormula interp val f1) (evalFormula interp val f2)
evalFormula interp val (Or f1 f2)  = disj (evalFormula interp val f1) (evalFormula interp val f2)
evalFormula interp val (Implies f1 f2) = implies (evalFormula interp val f1) (evalFormula interp val f2)
evalFormula interp val (Forall var f) =
  aggrA (domain interp) $ \x -> evalFormula interp (Map.insert var x val) f
evalFormula interp val (Exists var f) =
  aggrE (domain interp) $ \x -> evalFormula interp (Map.insert var x val) f

-- Helpers producing fuzzy degrees in (0,1)
sigmoid :: Double -> Double -> Double
sigmoid k z = 1 / (1 + exp (-k * z))

softGreater :: Double -> Double -> Double
softGreater x y = sigmoid 2 (x - y)

softLess :: Double -> Double -> Double
softLess x y = sigmoid 2 (y - x)

softEqual :: Double -> Double -> Double
softEqual x y = sigmoid 6 (-(abs (x - y)))

softEven :: Double -> Double
softEven x =
  let nearest = fromIntegral (2 * round (x / 2))
      dist = abs (x - nearest)
  in sigmoid 3 (1 - 2 * dist)

-- Example interpretation: integers 1-6 (like dice)
diceInterp :: Interpretation
diceInterp = Interpretation
  { domain = Finite [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  , functions = Map.fromList
      [ ("+", \[x, y] -> x + y)
      , ("*", \[x, y] -> x * y)
      ]
  , predicates = Map.fromList
      [ ("==", \[x, y] -> softEqual x y)
      , (">",  \[x, y] -> softGreater x y)
      , ("even", \[x]   -> softEven x)
      , ("closeTo", \[x, y] -> exp (-(x - y) * (x - y))) -- smooth proximity
      ]
  }

-- Test some formulas (dice)
diceFormula1 :: Formula
diceFormula1 = Exists "x" (Pred "even" [Var "x"])  -- ∃x. even(x)

diceFormula2 :: Formula
diceFormula2 = Forall "x" (Pred ">" [Var "x", Const 0])  -- ∀x. x > 0

diceFormula3 :: Formula
diceFormula3 = Exists "x" (Pred "==" [Var "x", Const 7])  -- ∃x. x = 7

-- Fuzzy: all dice values are close to 3.5 (geometric mean of closeness)
diceFormula4 :: Formula
diceFormula4 = Forall "x" (Pred "closeTo" [Var "x", Const 3.5])

-- Example interpretation: real interval [0,10]
realInterp :: Interpretation
realInterp = Interpretation
  { domain = RealInterval 0.0 10.0
  , functions = Map.fromList
      [ ("sin", \[x]   -> sin x)
      , ("cos", \[x]   -> cos x)
      , ("exp", \[x]   -> exp x)
      , ("+",   \[x,y] -> x + y)
      , ("*",   \[x,y] -> x * y)
      ]
  , predicates = Map.fromList
      [ ("<",       \[x, y] -> softLess x y)
      , (">",       \[x, y] -> softGreater x y)
      , ("==",      \[x, y] -> softEqual x y)
      , ("positive", \[x]    -> softGreater x 0)
      , ("near",     \[x,y]  -> exp (-((x - y)^2) / 2))
      ]
  }

-- Test formulas on real interval
realFormula1 :: Formula
realFormula1 = Exists "x" (Pred "near" [Appl "sin" [Var "x"], Const 0])

realFormula2 :: Formula
realFormula2 = Forall "x" (Pred "<" [Var "x", Const 15])  -- ∀x ∈ [0,10]. x < 15

realFormula3 :: Formula
realFormula3 = Forall "x" (Pred "near" [Appl "cos" [Var "x"], Const 1])

-- Example interpretation: natural numbers (countably infinite)
naturalsInterp :: Interpretation
naturalsInterp = Interpretation
  { domain = CountableInfinite [0.0, 1.0 ..]  -- 0, 1, 2, 3, ...
  , functions = Map.fromList
      [ ("+",    \[x, y] -> x + y)
      , ("*",    \[x, y] -> x * y)
      , ("succ", \[x]    -> x + 1)
      ]
  , predicates = Map.fromList
      [ ("==",  \[x, y] -> softEqual x y)
      , (">",   \[x, y] -> softGreater x y)
      , ("<",   \[x, y] -> softLess x y)
      , (">=",  \[x, y] -> softGreater x y)  -- soft >= approximated by softGreater
      , ("even", \[x]    -> softEven x)
      , ("closeToInt", \[x,y] -> exp (-abs (x - fromIntegral (round y))))
      ]
  }

-- Test formulas on naturals
natFormula1 :: Formula
natFormula1 = Exists "x" (Pred "even" [Var "x"])  -- ∃x ∈ ℕ. even(x)

natFormula2 :: Formula
natFormula2 = Forall "x" (Pred ">=" [Var "x", Const 0])  -- ∀x ∈ ℕ. x ≥ 0

natFormula3 :: Formula
natFormula3 = Exists "x" (Pred "==" [Var "x", Const 100])  -- ∃x ∈ ℕ. x = 100

main :: IO ()
main = do
  putStrLn "Truth values algebra defined"
  putStrLn "Domain types and aggregation defined"

  putStrLn "Syntax defined"
  putStrLn "Semantics defined"

  -- Dice examples
  let result1 = evalFormula diceInterp Map.empty diceFormula1
  putStrLn $ "∃x ∈ {1,2,3,4,5,6}. even(x) ≈ " ++ show result1

  let result2 = evalFormula diceInterp Map.empty diceFormula2
  putStrLn $ "∀x ∈ {1,2,3,4,5,6}. x > 0 ≈ " ++ show result2

  let result3 = evalFormula diceInterp Map.empty diceFormula3
  putStrLn $ "∃x ∈ {1,2,3,4,5,6}. x = 7 ≈ " ++ show result3

  -- Real interval examples
  let realResult1 = evalFormula realInterp Map.empty realFormula1
  putStrLn $ "∃x ∈ [0,10]. sin(x) ≈ 0 somewhere ≈ " ++ show realResult1

  let realResult2 = evalFormula realInterp Map.empty realFormula2
  putStrLn $ "∀x ∈ [0,10]. x < 15 ≈ " ++ show realResult2

  let realResult3 = evalFormula realInterp Map.empty realFormula3
  putStrLn $ "∀x ∈ [0,10]. cos(x) ≈ 1 ≈ " ++ show realResult3

  -- Naturals examples
  let natResult1 = evalFormula naturalsInterp Map.empty natFormula1
  putStrLn $ "∃x ∈ ℕ. even(x) ≈ " ++ show natResult1

  let natResult2 = evalFormula naturalsInterp Map.empty natFormula2
  putStrLn $ "∀x ∈ ℕ. x ≥ 0 ≈ " ++ show natResult2

  let natResult3 = evalFormula naturalsInterp Map.empty natFormula3
  putStrLn $ "∃x ∈ ℕ. x = 100 ≈ " ++ show natResult3

  return ()


