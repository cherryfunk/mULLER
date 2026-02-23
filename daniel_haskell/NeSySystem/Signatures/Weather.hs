module NeSySystem.Signatures.Weather (weatherSig) where

import Syntax (Signature (..))

-- | The Weather signature matching the DeepSeaProbLog example in the paper
-- Sorts: Worlds, Unit_Interval, Reals2, Humidity, Temperature
-- Funcs: data1, humid_detector, temperature_predictor
-- Rels: ==, <, >
-- MFuncs: bernoulli, normal
weatherSig :: Signature
weatherSig =
  Signature
    { sortDecls = ["Worlds", "Unit_Interval", "Reals2", "Humidity", "Temperature", "Reals"],
      funDecls =
        [ ("data1", [], "Worlds"),
          ("humid_detector", ["Worlds"], "Unit_Interval"),
          ("temperature_predictor", ["Worlds"], "Reals2")
        ],
      relDecls =
        [ ("==", ["Humidity", "Humidity"]),
          ("<", ["Temperature", "Temperature"]),
          (">", ["Temperature", "Temperature"])
        ],
      mFunDecls =
        [ ("bernoulli", ["Unit_Interval"], "Humidity"),
          ("normal", ["Reals2"], "Temperature")
        ],
      mRelDecls = []
    }
