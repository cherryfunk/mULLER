{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Non-logical interpretation: Weather domain (in DATA, with Giry monad)
module NonLogical.Interpretations.Weather where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..), tableLookup)
import NonLogical.Monads.Giry (Giry, categorical, normal)
import NonLogical.Signatures.WeatherSig

-- | Data instance (rows of the table)
worldsTable :: [WorldsRow]
worldsTable =
  [ WorldsRow "Berlin" 0.5 0.0 2.0,
    WorldsRow "Munich" 0.8 5.0 3.0,
    WorldsRow "Hamburg" 0.7 (-2.0) 1.5
  ]

-- | Row lookup (the "WHERE id = w" query)
lookupRow :: Worlds -> WorldsRow
lookupRow w = tableLookup worldId w worldsTable

instance WeatherSig DATA Giry where
  -- Sor (proving sorts are objects in DATA)
  worldsObj = Finite (map worldId worldsTable)
  humidDetectorObj = Reals
  temperaturePredictorObj = ProductObj Reals Reals

  -- Const
  data1 = "Berlin"

  -- Fun
  humid_detector = humidityPval . lookupRow
  temperature_predictor w = let r = lookupRow w in (tempMean r, tempStd r)

  -- mFun
  bernoulli p = categorical [(1 :: Int, p), (0 :: Int, 1.0 - p)]
  normalDist (mu, sigma) = normal mu sigma

  -- Rel
  eqInt x y = if x == y then v1 else v0
  ltDouble a b = 1.0 / (1.0 + exp (100.0 * (a - b)))
  gtDouble a b = 1.0 / (1.0 + exp (-100.0 * (a - b)))

-- mRel
