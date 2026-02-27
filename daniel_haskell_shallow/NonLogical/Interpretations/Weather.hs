-- | Interpretation 𝓘_Σ of WeatherSig in (DATA, Giry)
module NonLogical.Interpretations.Weather
  ( data1,
    humid_detector,
    temperature_predictor,
    bernoulli,
    normalDist,
  )
where

import NonLogical.Categories.DATA (tableLookup)
import NonLogical.Monads.Giry (Giry, categorical, normal)
import NonLogical.Signatures.WeatherSig (Worlds, WorldsRow (..))

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

-- | 𝓘(data1) : Const
data1 :: Worlds
data1 = "Berlin"

-- | 𝓘(humid_detector) : Fun
humid_detector :: Worlds -> Double
humid_detector = humidityPval . lookupRow

-- | 𝓘(temperature_predictor) : Fun
temperature_predictor :: Worlds -> (Double, Double)
temperature_predictor w = let r = lookupRow w in (tempMean r, tempStd r)

-- | 𝓘(bernoulli) : mFun
bernoulli :: Double -> Giry Int
bernoulli p = categorical [(1 :: Int, p), (0 :: Int, 1.0 - p)]

-- | 𝓘(normalDist) : mFun
normalDist :: (Double, Double) -> Giry Double
normalDist (mu, sigma) = normal mu sigma
