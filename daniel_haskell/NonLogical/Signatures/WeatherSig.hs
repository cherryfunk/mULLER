{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Non-logical signature: Weather domain
module NonLogical.Signatures.WeatherSig where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat)
import NonLogical.Categories.DATA (MonadOver)

-- | Table Sorts
type Worlds = String

-- | model Worlds {                       -- sort: Worlds
--     id             String  @id
--     humidityPval   Double              -- attribute: Worlds -> Double
--     tempMean       Double              -- attribute: Worlds -> Double
--     tempStd        Double              -- attribute: Worlds -> Double
--   }
data WorldsRow = WorldsRow
  { worldId :: Worlds,
    humidityPval :: Double,
    tempMean :: Double,
    tempStd :: Double
  }

class (TwoMonBLat Omega, MonadOver cat t) => WeatherSig cat t where
  -- Sor (all sorts used in this signature, proven to be in cat)
  worldsObj :: cat Worlds
  humidDetectorObj :: cat Double
  temperaturePredictorObj :: cat (Double, Double)

  -- Const
  data1 :: Worlds

  -- Fun
  humid_detector :: Worlds -> Double
  temperature_predictor :: Worlds -> (Double, Double)

  -- mFun
  bernoulli :: Double -> t Int
  normalDist :: (Double, Double) -> t Double

  -- Rel
  eqInt :: Int -> Int -> Omega
  ltDouble :: Double -> Double -> Omega
  gtDouble :: Double -> Double -> Omega

-- mRel
