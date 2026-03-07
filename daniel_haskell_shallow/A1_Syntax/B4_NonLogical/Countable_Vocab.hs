{-# LANGUAGE AllowAmbiguousTypes #-}

module A1_Syntax.B4_NonLogical.Countable_Vocab where

-- | Non-Logical Vocabulary for the Countable sets domain.

-- | Signature:
class Countable_Vocab m where
  -- mFun (Kleisli):
  drawInt :: m Int
  drawStr :: m String
  drawLazy :: m Int
  drawHeavy :: m Int
