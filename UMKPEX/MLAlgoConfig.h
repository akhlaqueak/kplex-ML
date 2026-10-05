#ifndef MLALGOCONFIG_H
#define MLALGOCONFIG_H

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
using std::ifstream;
using std::map;
using std::string;

/**
 * Loads ml_algo_config.json (or whatever -ml-algo-config points to): the per-label
 * CommandLine overrides for the 24 Scheme 1 ML algorithm classes
 * (restructured_pipeline/src/features.py::y_columns), consumed by
 * ml_label_to_commandline() in main.cpp. Edit ml_algo_config.json to fill these in --
 * not this file.
 *
 * File format: a JSON object mapping each label to an override object, using the
 * same keys/values a manual -race spec accepts (see apply_race_override in main.cpp):
 *   framework : default | oriented
 *   algorithm : UMKP+ | kPEX   (plain "UMKP" isn't a valid value yet -- see
 *               CommandLine::valid_algorithms() in Command.h if 8 of the 24 labels
 *               need it added as a real solver codepath)
 *   branching : Adaptive-Br | Pivot-Br | S-Br | R-Br | SR-Br | Binary-Br | BinaryM-Br
 *   bounding  : None | S-Bound | R-Bound | SR-Bound | AltRB
 *   ctcp      : CTCP | CF-CTCP
 *   UBR2      : "true" | "false"
 * A label with no entries for a given key inherits that field from the base config
 * (whatever -algorithm/-branching/etc. were passed on the command line, or their
 * defaults in Command.h). Example:
 *   "UMKP+-oriented-B-bnb": {"algorithm": "UMKP+", "framework": "oriented", "branching": "Binary-Br"}
 *
 * This is a hand-written parser for exactly that one shape -- a two-level object of
 * string:string pairs, no arrays/numbers/booleans/nesting -- rather than pulling in a
 * general JSON library for a 24-entry config file.
 */

namespace ml_algo_config_detail
{
    inline void skip_ws(const string &s, size_t &i)
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            i++;
    }

    inline string parse_string(const string &s, size_t &i)
    {
        if (i >= s.size() || s[i] != '"')
            throw std::runtime_error("ml_algo_config: expected '\"' at position " + std::to_string(i));
        i++;
        string out;
        while (i < s.size() && s[i] != '"')
        {
            char c = s[i];
            if (c == '\\' && i + 1 < s.size())
            {
                char next = s[i + 1];
                switch (next)
                {
                case 'n':
                    out += '\n';
                    break;
                case 't':
                    out += '\t';
                    break;
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                default:
                    out += next;
                    break;
                }
                i += 2;
            }
            else
            {
                out += c;
                i++;
            }
        }
        if (i >= s.size())
            throw std::runtime_error("ml_algo_config: unterminated string");
        i++;
        return out;
    }

    inline map<string, string> parse_override_object(const string &s, size_t &i, const string &label)
    {
        map<string, string> overrides;
        if (i >= s.size() || s[i] != '{')
            throw std::runtime_error("ml_algo_config: expected '{' for label '" + label + "'");
        i++;
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}')
        {
            i++;
            return overrides;
        }
        while (true)
        {
            skip_ws(s, i);
            string key = parse_string(s, i);
            skip_ws(s, i);
            if (i >= s.size() || s[i] != ':')
                throw std::runtime_error("ml_algo_config: expected ':' after key '" + key + "' in label '" + label + "'");
            i++;
            skip_ws(s, i);
            string value = parse_string(s, i);
            overrides[key] = value;
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',')
            {
                i++;
                continue;
            }
            break;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != '}')
            throw std::runtime_error("ml_algo_config: expected '}' to close label '" + label + "'");
        i++;
        return overrides;
    }

    inline map<string, map<string, string>> parse_label_config(const string &text)
    {
        map<string, map<string, string>> result;
        size_t i = 0;
        skip_ws(text, i);
        if (i >= text.size() || text[i] != '{')
            throw std::runtime_error("ml_algo_config: expected top-level '{'");
        i++;
        skip_ws(text, i);
        if (i < text.size() && text[i] == '}')
            return result;

        while (true)
        {
            skip_ws(text, i);
            string label = parse_string(text, i);
            skip_ws(text, i);
            if (i >= text.size() || text[i] != ':')
                throw std::runtime_error("ml_algo_config: expected ':' after label '" + label + "'");
            i++;
            skip_ws(text, i);
            result[label] = parse_override_object(text, i, label);

            skip_ws(text, i);
            if (i < text.size() && text[i] == ',')
            {
                i++;
                continue;
            }
            break;
        }
        skip_ws(text, i);
        if (i >= text.size() || text[i] != '}')
            throw std::runtime_error("ml_algo_config: expected top-level '}'");
        return result;
    }
} // namespace ml_algo_config_detail

/**
 * @brief Returns the parsed label->overrides map, reading `path` only once per
 * process (cached in a function-local static).
 */
inline const map<string, map<string, string>> &load_ml_algo_config(const string &path)
{
    static map<string, map<string, string>> cached;
    static bool loaded = false;
    if (!loaded)
    {
        ifstream in(path);
        if (!in.is_open())
            throw std::runtime_error("Could not open ML algorithm config file: " + path);
        std::stringstream buf;
        buf << in.rdbuf();
        cached = ml_algo_config_detail::parse_label_config(buf.str());
        loaded = true;
    }
    return cached;
}

#endif // MLALGOCONFIG_H
