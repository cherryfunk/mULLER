{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Non-logical interpretation: Weather domain (in DATA, with Giry monad)
module NonLogical.Interpretations.Weather
  ( humid',
    temp',
    eqI',
    ltD',
    gtD',
  )
where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..), tableLookup)
import NonLogical.Monads.Giry (Giry, categorical, normal)
import NonLogical.Signatures.WeatherSig
import TypedSyntax

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
  worldsObj = Finite (map worldId worldsTable)
  humidDetectorObj = Reals
  temperaturePredictorObj = ProductObj Reals Reals
  data1 = "Berlin"
  humid_detector = humidityPval . lookupRow
  temperature_predictor w = let r = lookupRow w in (tempMean r, tempStd r)
  bernoulli p = categorical [(1 :: Int, p), (0 :: Int, 1.0 - p)]
  normalDist (mu, sigma) = normal mu sigma
  eqInt x y = if x == y then v1 else v0
  ltDouble a b = 1.0 / (1.0 + exp (100.0 * (a - b)))
  gtDouble a b = 1.0 / (1.0 + exp (-100.0 * (a - b)))

-- | Formula-ready symbols
humid' :: Term (Giry Int)
humid' = con (bernoulli @DATA @Giry) $$ (con (humid_detector @DATA @Giry) $$ con (data1 @DATA @Giry))

temp' :: Term (Giry Double)
temp' = con (normalDist @DATA @Giry) $$ (con (temperature_predictor @DATA @Giry) $$ con (data1 @DATA @Giry))

eqI' :: Term Int -> Term Int -> Formula Omega
eqI' x y = rel (con (eqInt @DATA @Giry) $$ x $$ y)

ltD' :: Term Double -> Term Double -> Formula Omega
ltD' x y = rel (con (ltDouble @DATA @Giry) $$ x $$ y)

gtD' :: Term Double -> Term Double -> Formula Omega
gtD' x y = rel (con (gtDouble @DATA @Giry) $$ x $$ y)
