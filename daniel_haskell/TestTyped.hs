{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TestTyped where

import qualified Data.Map as Map
import Logical.Interpretations.Boolean (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import TypedSemantics
import TypedSyntax

-- | A simple test formula using the Boolean interpretation
testFormula :: Formula Omega
testFormula =
  let boolEmbed b = if b then (v1 :: Omega) else (v0 :: Omega)
   in BinConn
        wedge
        (Rel (Con boolEmbed `Fun` (Con even `Fun` Con (3 :: Int))))
        (Rel (Con boolEmbed `Fun` (Con ((==) @Int) `Fun` Con (3 :: Int) `Fun` Con (4 :: Int))))

testRun :: IO ()
testRun = do
  let result = evalFormula @[] @Omega @Omega Map.empty testFormula
  putStrLn $ "testFormula = " ++ show result
