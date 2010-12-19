/**
 *          Copyright Matthias Walter 2010.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 **/

#include "../config.h"
#include <fstream>
#include <iomanip>

#include "total_unimodularity.hpp"
#include "matroid_decomposition.hpp"

#include "gen_generic.hpp"
#include "gen_cycle_violator.hpp"
#include "gen_network.hpp"
#include "gen_random.hpp"

bool extract_option(char c, bool& randomize, bool& sign, tu::log_level& level, bool& help)
{
  if (c == 's')
    sign = true;
  else if (c == 'r')
    randomize = true;
  else if (c == 'h')
    help = true;
  else if (c == 'q')
    level = tu::LOG_QUIET;
  else if (c == 'v')
    level = tu::LOG_VERBOSE;
  else
    return false;

  return true;
}

int main(int argc, char** argv)
{
  /// Possible parameters
  size_t non_option = 0;
  tu::log_level level = tu::LOG_VERBOSE;
  bool randomize = false;
  bool help = false;
  bool sign = false;
  char type = 0;
  size_t width = 0;
  size_t height = std::numeric_limits <size_t>::max();

  bool options_done = false;
  for (int a = 1; a < argc; ++a)
  {
    const std::string current = argv[a];

    if (!options_done)
    {
      if (current == std::string("--"))
      {
        options_done = true;
        continue;
      }
      else if (current.size() > 0 && current[0] == '-')
      {
        for (size_t i = 1; i < current.size(); ++i)
        {
          if (!extract_option(current[i], randomize, sign, level, help))
          {
            std::cerr << "Unknown option: -" << current[i] << "\nSee " << argv[0] << " -h for usage." << std::endl;
            return EXIT_FAILURE;
          }
        }
        continue;
      }
    }

    if (non_option == 0)
    {
      if (current == "r" || current == "rnd" || current == "random")
        type = 'r';
      else if (current == "n" || current == "net" || current == "network")
        type = 'n';
      else if (current == "c" || current == "cv" || current == "cycle" || current == "cycle-violator")
        type = 'c';
      else
        type = ' ';
    }
    else if (non_option == 1)
    {
      std::stringstream ss(current);
      ss >> height;
      if (ss.fail() || ss.good())
      {
        std::cerr << "Unable to parse height \"" << current << "\"! See " << argv[0] << " -h for usage." << std::endl;
        return EXIT_FAILURE;
      }
      width = height;
    }
    else if (non_option == 2)
    {
      std::stringstream ss(current);
      ss >> width;
      if (ss.fail() || ss.good())
      {
        std::cerr << "Unable to parse width \"" << current << "\"! See " << argv[0] << " -h for usage." << std::endl;
        return EXIT_FAILURE;
      }
    }
    else
    {
      std::cerr << "Runaway argument \"" << current << "\"!\nSee " << argv[0] << " -h for usage." << std::endl;
      return EXIT_FAILURE;
    }

    non_option++;
  }

  if (height == std::numeric_limits <size_t>::max())
  {
    std::cerr << "Size of matrix not given!\nSee " << argv[0] << " -h for usage." << std::endl;
    return EXIT_FAILURE;
  }

  if (help)
  {
    std::cerr << "Usage: " << argv[0] << " [OPTIONS] [--] TYPE HEIGHT [WIDTH]\n";
    std::cerr << "Types:\n";
    std::cerr << "  r Generates a matrix with a random entry at each position (prob. not t.u.).\n";
    std::cerr << "  n Generates a network matrix (t.u.).\n";
    std::cerr << "  v Generates a cycle-based violator matrix (only height=width in 2Z + 1).\n";
    std::cerr << "Options:\n";
    std::cerr << " -p Use pivots (to hide violator or randomize).\n";
    std::cerr << " -h Shows a help message.\n";
    std::cerr << " -v Prints information while generating the matrix (default).\n";
    std::cerr << " -q Prints nothing except the matrix.\n";
    std::cerr << "Omitting the WIDTH parameter sets the width equal to the height.\n";
    std::cerr << std::flush;
    return EXIT_SUCCESS;
  }

  matrix_generator* generator = NULL;
  if (type == 'r')
  {
    generator = new random_matrix_generator(height, width, level);
  }
  else if (type == 'n')
  {
    generator = new network_matrix_generator(height, width, level);
  }
  else if (type == 'c')
  {
    if (width != height)
    {
      std::cerr << "Cycle-violator matrices must be square matrices!\nSee " << argv[0] << " -h for usage." << std::endl;
      return EXIT_FAILURE;
    }
    if (height % 2 == 0)
    {
      std::cerr << "Cycle-violator matrices must be of odd size!\nSee " << argv[0] << " -h for usage." << std::endl;
      return EXIT_FAILURE;
    }

    generator = new cycle_violator_matrix_generator(height, level);
  }
  else
  {
    std::cerr << "No or invalid algorithm given!\nSee " << argv[0] << " -h for usage." << std::endl;
    return EXIT_FAILURE;
  }

  assert(generator);

  generator->generate();

  if (randomize)
  {
    if (level != tu::LOG_QUIET)
      std::cerr << "Randomizing the resulting matrix..." << std::flush;
    generator->randomize();
    if (level != tu::LOG_QUIET)
      std::cerr << " done." << std::endl;
  }
  if (sign)
  {
    if (level != tu::LOG_QUIET)
      std::cerr << "Making the resulting matrix signed..." << std::flush;
    generator->sign();
    if (level != tu::LOG_QUIET)
      std::cerr << " done." << std::endl;
  }

  generator->print();

  return EXIT_SUCCESS;
}
