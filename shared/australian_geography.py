# shared/australian_geography.py
"""
Australian geographic data for city-state mapping and validation
"""

# Comprehensive city to state mapping for Australia
AUSTRALIAN_CITIES = {
    # New South Wales
    'sydney': 'New South Wales',
    'newcastle': 'New South Wales',
    'wollongong': 'New South Wales',
    'central coast': 'New South Wales',
    'maitland': 'New South Wales',
    'parramatta': 'New South Wales',
    'blacktown': 'New South Wales',
    'liverpool': 'New South Wales',
    'penrith': 'New South Wales',
    'campbelltown': 'New South Wales',
    'bankstown': 'New South Wales',
    'sutherland': 'New South Wales',
    'chatswood': 'New South Wales',
    'bondi': 'New South Wales',
    'manly': 'New South Wales',
    'coffs harbour': 'New South Wales',
    'wagga wagga': 'New South Wales',
    'albury': 'New South Wales',
    'tamworth': 'New South Wales',
    'orange': 'New South Wales',
    'dubbo': 'New South Wales',
    'broken hill': 'New South Wales',
    'lismore': 'New South Wales',
    'port macquarie': 'New South Wales',

    # Victoria
    'melbourne': 'Victoria',
    'geelong': 'Victoria',
    'ballarat': 'Victoria',
    'bendigo': 'Victoria',
    'shepparton': 'Victoria',
    'mildura': 'Victoria',
    'warrnambool': 'Victoria',
    'traralgon': 'Victoria',
    'wangaratta': 'Victoria',
    'horsham': 'Victoria',
    'sale': 'Victoria',
    'bairnsdale': 'Victoria',
    'portland': 'Victoria',
    'hamilton': 'Victoria',
    'colac': 'Victoria',
    'swan hill': 'Victoria',
    'benalla': 'Victoria',
    'ararat': 'Victoria',
    'stawell': 'Victoria',
    'echuca': 'Victoria',
    'wonthaggi': 'Victoria',
    'castlemaine': 'Victoria',
    'kyneton': 'Victoria',
    'healesville': 'Victoria',
    'warragul': 'Victoria',

    # Queensland
    'brisbane': 'Queensland',
    'gold coast': 'Queensland',
    'sunshine coast': 'Queensland',
    'townsville': 'Queensland',
    'cairns': 'Queensland',
    'toowoomba': 'Queensland',
    'rockhampton': 'Queensland',
    'mackay': 'Queensland',
    'bundaberg': 'Queensland',
    'hervey bay': 'Queensland',
    'gladstone': 'Queensland',
    'maryborough': 'Queensland',
    'mount isa': 'Queensland',
    'gympie': 'Queensland',
    'caboolture': 'Queensland',
    'ipswich': 'Queensland',
    'logan': 'Queensland',
    'redcliffe': 'Queensland',
    'maroochydore': 'Queensland',

    # Western Australia
    'perth': 'Western Australia',
    'fremantle': 'Western Australia',
    'mandurah': 'Western Australia',
    'bunbury': 'Western Australia',
    'rockingham': 'Western Australia',
    'joondalup': 'Western Australia',
    'albany': 'Western Australia',
    'kalgoorlie': 'Western Australia',
    'geraldton': 'Western Australia',
    'broome': 'Western Australia',
    'busselton': 'Western Australia',
    'karratha': 'Western Australia',
    'port hedland': 'Western Australia',

    # South Australia
    'adelaide': 'South Australia',
    'mount gambier': 'South Australia',
    'whyalla': 'South Australia',
    'murray bridge': 'South Australia',
    'port augusta': 'South Australia',
    'port pirie': 'South Australia',
    'port lincoln': 'South Australia',
    'victor harbor': 'South Australia',
    'gawler': 'South Australia',

    # Tasmania
    'hobart': 'Tasmania',
    'launceston': 'Tasmania',
    'devonport': 'Tasmania',
    'burnie': 'Tasmania',
    'ulverstone': 'Tasmania',
    'kingston': 'Tasmania',

    # Northern Territory
    'darwin': 'Northern Territory',
    'alice springs': 'Northern Territory',
    'palmerston': 'Northern Territory',
    'katherine': 'Northern Territory',

    # Australian Capital Territory
    'canberra': 'Australian Capital Territory',
    'queanbeyan': 'Australian Capital Territory',
}

# State abbreviations
STATE_ABBREVIATIONS = {
    'nsw': 'New South Wales',
    'vic': 'Victoria',
    'qld': 'Queensland',
    'wa': 'Western Australia',
    'sa': 'South Australia',
    'tas': 'Tasmania',
    'nt': 'Northern Territory',
    'act': 'Australian Capital Territory',
}

# Reverse mapping
ABBREVIATION_TO_STATE = {v: k for k, v in STATE_ABBREVIATIONS.items()}

# Major regions/areas within states
REGIONS_TO_STATE = {
    'sydney cbd': 'New South Wales',
    'melbourne cbd': 'Victoria',
    'brisbane cbd': 'Queensland',
    'greater sydney': 'New South Wales',
    'greater melbourne': 'Victoria',
    'greater brisbane': 'Queensland',
    'hunter valley': 'New South Wales',
    'illawarra': 'New South Wales',
    'gold coast': 'Queensland',
    'sunshine coast': 'Queensland',
    'great ocean road': 'Victoria',
    'yarra valley': 'Victoria',
    'mornington peninsula': 'Victoria',
    'margaret river': 'Western Australia',
    'barossa valley': 'South Australia',
    'fleurieu peninsula': 'South Australia',
}


def get_state_from_location(location: str) -> str:
    """
    Get state from a location string (city, region, or address)

    Args:
        location: Location string to parse

    Returns:
        State name or None if not found
    """
    if not location:
        return None

    location_lower = location.lower().strip()

    # Check direct city match
    for city, state in AUSTRALIAN_CITIES.items():
        if city in location_lower:
            return state

    # Check regions
    for region, state in REGIONS_TO_STATE.items():
        if region in location_lower:
            return state

    # Check state names directly
    states = ['new south wales', 'victoria', 'queensland', 'western australia',
              'south australia', 'tasmania', 'northern territory', 'australian capital territory']
    for state in states:
        if state in location_lower:
            return state

    # Check abbreviations
    for abbrev, state in STATE_ABBREVIATIONS.items():
        # Look for abbreviation with word boundaries
        if f' {abbrev}' in location_lower or f',{abbrev}' in location_lower or location_lower.endswith(abbrev):
            return state

    return None


def validate_location_in_state(location: str, target_states: list) -> bool:
    """
    Check if a location is in any of the target states

    Args:
        location: Location string to check
        target_states: List of target state names

    Returns:
        True if location is in one of the target states
    """
    if not location or not target_states:
        return False

    detected_state = get_state_from_location(location)
    if not detected_state:
        return False

    # Normalize target states
    normalized_targets = []
    for state in target_states:
        state_lower = state.lower().strip()
        # Check if it's an abbreviation
        if state_lower in STATE_ABBREVIATIONS:
            normalized_targets.append(STATE_ABBREVIATIONS[state_lower])
        else:
            normalized_targets.append(state)

    return detected_state in normalized_targets


def get_cities_in_state(state: str, major_only: bool = True) -> list:
    """
    Get list of cities in a state

    Args:
        state: State name
        major_only: If True, return only major cities

    Returns:
        List of city names
    """
    # Normalize state name
    state_normalized = state
    if state.lower() in STATE_ABBREVIATIONS:
        state_normalized = STATE_ABBREVIATIONS[state.lower()]

    cities = [city for city, s in AUSTRALIAN_CITIES.items() if s == state_normalized]

    if major_only:
        # Define major cities per state
        major_cities = {
            'New South Wales': ['sydney', 'newcastle', 'wollongong', 'parramatta'],
            'Victoria': ['melbourne', 'geelong', 'ballarat', 'bendigo'],
            'Queensland': ['brisbane', 'gold coast', 'sunshine coast', 'townsville', 'cairns'],
            'Western Australia': ['perth', 'fremantle', 'mandurah'],
            'South Australia': ['adelaide', 'mount gambier'],
            'Tasmania': ['hobart', 'launceston'],
            'Northern Territory': ['darwin', 'alice springs'],
            'Australian Capital Territory': ['canberra'],
        }

        if state_normalized in major_cities:
            return [c for c in cities if c in major_cities[state_normalized]]

    return cities