module NeSySystem.Signatures.Weather (weatherSig) where

import Syntax (Signature (..))

-- | The Weather signature
-- Sorts: Humidity, Temperature
-- Funcs: humid_detector, temperature_predictor
-- Rels: ==, <, >
-- MFuncs: bernoulli, normal
weatherSig :: Signature
weatherSig =
  Signature
    { sortDecls = ["Humidity", "Temperature"],
      funDecls =
        [ ("humid_detector", ["Humidity"], "Temperature"),
          ("temperature_predictor", ["Humidity"], "Temperature")
        ],
      relDecls =
        [ ("==", ["Humidity", "Humidity"]),
          ("<", ["Temperature", "Temperature"]),
          (">", ["Temperature", "Temperature"])
        ],
      mFunDecls =
        [ ("bernoulli", ["Temperature"], "Humidity"),
          ("normal", ["Temperature"], "Temperature")
        ],
      mRelDecls = []
    }
