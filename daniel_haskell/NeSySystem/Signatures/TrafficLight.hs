module NeSySystem.Signatures.TrafficLight (trafficSig) where

import Syntax (Signature (..))

-- | The Traffic Light signature
-- Sorts: Light, Drive
-- Relations: ==, eval
trafficSig :: Signature
trafficSig =
  Signature
    { sortDecls = ["Light", "Drive"],
      funDecls = [],
      relDecls =
        [ ("==", ["Light", "Light"]),
          ("eval", ["Drive"])
        ]
    }
