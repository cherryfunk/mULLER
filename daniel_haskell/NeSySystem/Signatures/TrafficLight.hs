module NeSySystem.Signatures.TrafficLight (trafficSig) where

import Syntax (Signature (..))

-- | The Traffic Light signature
-- Sorts: Light, Drive
-- Rels: ==, eval
-- MFuncs: light, driveF
-- MRels: driveP
trafficSig :: Signature
trafficSig =
  Signature
    { sortDecls = ["Light", "Drive"],
      funDecls = [],
      relDecls =
        [ ("==", ["Light", "Light"]),
          ("eval", ["Drive"])
        ],
      mFunDecls =
        [ ("light", [], "Light"),
          ("driveF", ["Light"], "Drive")
        ],
      mRelDecls =
        [ ("driveP", ["Light"])
        ]
    }
