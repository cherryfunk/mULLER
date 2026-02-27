{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Non-logical signature: Countable sets domain
module NonLogical.Signatures.CountableSig where

import Data.Kind (Type)
import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat)
import NonLogical.Categories.DATA (MonadOver)

class (TwoMonBLat Omega, MonadOver cat t) => CountableSig (cat :: Type -> Type) t where
  -- Sor

  -- Const

  -- Fun

  -- mFun
  drawInt :: t Int
  drawStr :: t String
  drawLazy :: t Int
  drawHeavy :: t Int

  -- Rel
  gt3 :: Int -> Omega
  startsTT :: String -> Omega
  isEven :: Int -> Omega
  isAnything :: Int -> Omega

-- mRel
