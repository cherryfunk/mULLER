module Formulas (loadFormulas) where

import Data.List (isPrefixOf, stripPrefix)
import qualified Data.Map as Map

-- | Reads a LaTeX file and extracts lines of the form:
--   \item \defFormula{NAME}{FORMULA}
--   Returns a Map from NAME to FORMULA.
loadFormulas :: FilePath -> IO (Map.Map String String)
loadFormulas path = do
  content <- readFile path
  return $ foldl extract Map.empty (lines content)
  where
    -- Extract formula by balancing curly braces
    extract m line =
      case stripPrefix "\\item \\defFormula{" (dropWhile (== ' ') line) of
        Just rest ->
          let (name, rest') = break (== '}') rest
           in case stripPrefix "}{" rest' of
                Just fRest ->
                  let formula = extractBraced 0 "" fRest
                   in Map.insert name formula m
                Nothing -> m
        Nothing -> m

    extractBraced :: Int -> String -> String -> String
    extractBraced _ acc [] = reverse acc
    extractBraced depth acc (c : cs)
      | c == '{' = extractBraced (depth + 1) (c : acc) cs
      | c == '}' =
          if depth == 0
            then reverse acc
            else extractBraced (depth - 1) (c : acc) cs
      | otherwise = extractBraced depth (c : acc) cs
