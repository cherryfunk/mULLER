{-# LANGUAGE AllowAmbiguousTypes #-}

module A1_Syntax.B4_NonLogical.Weather_Vocab where

-- | Non-Logical Vocabulary for the Weather domain.

-- | Sorts:
type Worlds = String

-- | Data schema:
data WorldsRow = WorldsRow
  { worldId :: Worlds,
    humidityPval :: Double,
    tempMean :: Double,
    tempStd :: Double
  }

-- | Signature:
class Weather_Vocab m where
  -- Con:
  data1 :: Worlds
  data2 :: Worlds
  data3 :: Worlds

  -- Fun (Tarski):
  humidDetect :: Worlds -> Double
  tempPredict :: Worlds -> (Double, Double)

  -- mFun (Kleisli):
  bernoulli :: Double -> m Int
  normalDist :: (Double, Double) -> m Double
