-- | Weather domain — Signature + Interpretation
module NonLogical.Interpretations.Weather where

import NonLogical.Categories.DATA (tableLookup)
import NonLogical.Monads.Giry (Giry, categorical, normal)

--------------------------------------------------------------------------------
-- Σ: Non-Logical Vocabulary (sorts, data schema)
--------------------------------------------------------------------------------

-- | Sor
type Worlds = String

-- | Data layout (analogous to a database schema)
data WorldsRow = WorldsRow
  { worldId :: Worlds,
    humidityPval :: Double,
    tempMean :: Double,
    tempStd :: Double
  }

-- | Con:  data1       :: Worlds
-- | Fun:  humidDetect :: Worlds -> Double
-- |       tempPredict :: Worlds -> (Double, Double)
-- | mFun: bernoulli   :: Double -> Giry Int
-- |       normalDist  :: (Double, Double) -> Giry Double

--------------------------------------------------------------------------------
-- 𝓘: Interpretation and their Syntctic Type Declarations
--------------------------------------------------------------------------------

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

-- | 𝓘(data1) : Con
data1 :: Worlds
data1 = "Berlin"

-- | 𝓘(humidDetect) : Fun
humidDetect :: Worlds -> Double
humidDetect = humidityPval . lookupRow

-- | 𝓘(tempPredict) : Fun
tempPredict :: Worlds -> (Double, Double)
tempPredict w = let r = lookupRow w in (tempMean r, tempStd r)

-- | 𝓘(bernoulli) : mFun
bernoulli :: Double -> Giry Int
bernoulli p = categorical [(1 :: Int, p), (0 :: Int, 1.0 - p)]

-- | 𝓘(normalDist) : mFun
normalDist :: (Double, Double) -> Giry Double
normalDist (mu, sigma) = normal mu sigma
