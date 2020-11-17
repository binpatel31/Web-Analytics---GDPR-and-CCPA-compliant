function fingerprint_browser() {
    "use strict";
    var strOnError, strUserAgent, numVersion, strBrowser, strOut;

    strOnError = "Error";
    strUserAgent = null;
    numVersion = null;
    strBrowser  = null;
    strOut = null;

    try {
        strUserAgent = navigator.userAgent.toLowerCase();
        if (/msie (\d+\.\d+);/.test(strUserAgent)) {
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            if (strUserAgent.indexOf("trident/6") > -1) {
                numVersion = 10;
            }
            if (strUserAgent.indexOf("trident/5") > -1) {
                numVersion = 9;
            }
            if (strUserAgent.indexOf("trident/4") > -1) {
                numVersion = 8;
            }
            strBrowser = "Internet Explorer " + numVersion;
        } else if (strUserAgent.indexOf("trident/7") > -1) { //IE 11+ gets rid of the legacy 'MSIE' in the user-agent string;
            numVersion = 11;
            strBrowser = "Internet Explorer " + numVersion;
        }  else if (/firefox[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Firefox/x.x or Firefox x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Firefox " + numVersion;
        } else if (/opera[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Opera/x.x or Opera x.x (ignoring remaining decimal places);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Opera " + numVersion;
        } else if (/chrome[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Chrome/x.x or Chrome x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Chrome " + numVersion;
        } else if (/version[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Version/x.x or Version x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Safari " + numVersion;
        } else if (/rv[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for rv/x.x or rv x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Mozilla " + numVersion;
        } else if (/mozilla[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Mozilla/x.x or Mozilla x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Mozilla " + numVersion;
        } else if (/binget[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for BinGet/x.x or BinGet x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (BinGet) " + numVersion;
        } else if (/curl[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Curl/x.x or Curl x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (cURL) " + numVersion;
        } else if (/java[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Java/x.x or Java x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (Java) " + numVersion;
        } else if (/libwww-perl[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for libwww-perl/x.x or libwww-perl x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (libwww-perl) " + numVersion;
        } else if (/microsoft url control -[\s](\d+\.\d+)/.test(strUserAgent)) { //test for Microsoft URL Control - x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (Microsoft URL Control) " + numVersion;
        } else if (/peach[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for Peach/x.x or Peach x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (Peach) " + numVersion;
        } else if (/php[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for PHP/x.x or PHP x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (PHP) " + numVersion;
        } else if (/pxyscand[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for pxyscand/x.x or pxyscand x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (pxyscand) " + numVersion;
        } else if (/pycurl[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for pycurl/x.x or pycurl x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (PycURL) " + numVersion;
        } else if (/python-urllib[\/\s](\d+\.\d+)/.test(strUserAgent)) { //test for python-urllib/x.x or python-urllib x.x (ignoring remaining digits);
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Library (Python URLlib) " + numVersion;
        } else if (/appengine-google/.test(strUserAgent)) { //test for AppEngine-Google;
            numVersion = Number(RegExp.$1); // capture x.x portion and store as a number
            strBrowser = "Cloud (Google AppEngine) " + numVersion;
        } else {
            strBrowser = "Unknown";
        }
        strOut = strBrowser;
        return strOut;
    } catch (err) {
        return strOnError;
    }
}


function fingerprint_connection() {
    "use strict";
    var strOnError, strConnection, strOut;

    strOnError = "N/A";
    strConnection = null;
    strOut = null;

    try {
		// only on android
        strConnection = navigator.connection.type;
        strOut = strConnection;
    } catch (err) {
		// return N/A if navigator.connection object does not apply to this device
        return strOnError;
    }
    return strOut;
}


function fingerprint_cookie() {
    "use strict";
    var strOnError, bolCookieEnabled, bolOut;

    strOnError = "Error";
    bolCookieEnabled = null;
    bolOut = null;

    try {
        bolCookieEnabled = (navigator.cookieEnabled) ? true : false;

        //if not IE4+ nor NS6+
        if (typeof navigator.cookieEnabled === "undefined" && !bolCookieEnabled) {
            document.cookie = "testcookie";
            bolCookieEnabled = (document.cookie.indexOf("testcookie") !== -1) ? true : false;
        }
        bolOut = bolCookieEnabled;
        return bolOut;
    } catch (err) {
        return strOnError;
    }
}



function fingerprint_display() {
    "use strict";
    var strSep, strPair, strOnError, strScreen, strDisplay, strOut;

    strSep = "|";
    strPair = "=";
    strOnError = "Error";
    strScreen = null;
    strDisplay = null;
    strOut = null;

    try {
        strScreen = window.screen;
        if (strScreen) {
            strDisplay = strScreen.colorDepth + strSep + strScreen.width + strSep + strScreen.height;
        }
        strOut = strDisplay;
        return strOut;
    } catch (err) {
        return strOnError;
    }
}



function fingerprint_language() {
    "use strict";
    var strSep, strPair, strOnError, strLang, strTypeLng, strTypeBrLng, strTypeSysLng, strTypeUsrLng, strOut;

    strSep = "|";
    strPair = "=";
    strOnError = "Error";
    strLang = null;
    strTypeLng = null;
    strTypeBrLng = null;
    strTypeSysLng = null;
    strTypeUsrLng = null;
    strOut = null;

    try {
        strTypeLng = typeof (navigator.language);
        strTypeBrLng = typeof (navigator.browserLanguage);
        strTypeSysLng = typeof (navigator.systemLanguage);
        strTypeUsrLng = typeof (navigator.userLanguage);

        if (strTypeLng !== "undefined") {
            strLang = "lang" + strPair + navigator.language + strSep;
        } else if (strTypeBrLng !== "undefined") {
            strLang = "lang" + strPair + navigator.browserLanguage + strSep;
        } else {
            strLang = "lang" + strPair + strSep;
        }
        if (strTypeSysLng !== "undefined") {
            strLang += "syslang" + strPair + navigator.systemLanguage + strSep;
        } else {
            strLang += "syslang" + strPair + strSep;
        }
        if (strTypeUsrLng !== "undefined") {
            strLang += "userlang" + strPair + navigator.userLanguage;
        } else {
            strLang += "userlang" + strPair;
        }
        strOut = strLang;
        return strOut;
    } catch (err) {
        return strOnError;
    }
}


function fingerprint_os() {
    "use strict";
    var strSep, strOnError, strUserAgent, strPlatform, strOS, strOSBits, strOut;

    strSep = "|";
    strOnError = "Error";
    strUserAgent = null;
    strPlatform = null;
    strOS = null;
    strOSBits = null;
    strOut = null;

    try {
        strUserAgent = navigator.userAgent.toLowerCase();
        strPlatform = navigator.platform.toLowerCase();
        if (strUserAgent.indexOf("windows nt 10.0") !== -1) {
            strOS = "Windows 10";
        } else if (strUserAgent.indexOf("windows nt 6.3") !== -1) {
            strOS = "Windows 8.1";
        } else if (strUserAgent.indexOf("windows nt 6.2") !== -1) {
            strOS = "Windows 8";
        } else if (strUserAgent.indexOf("windows nt 6.1") !== -1) {
            strOS = "Windows 7";
        } else if (strUserAgent.indexOf("windows nt 6.0") !== -1) {
            strOS = "Windows Vista/Windows Server 2008";
        } else if (strUserAgent.indexOf("windows nt 5.2") !== -1) {
            strOS = "Windows XP x64/Windows Server 2003";
        } else if (strUserAgent.indexOf("windows nt 5.1") !== -1) {
            strOS = "Windows XP";
        } else if (strUserAgent.indexOf("windows nt 5.01") !== -1) {
            strOS = "Windows 2000, Service Pack 1 (SP1)";
        } else if (strUserAgent.indexOf("windows xp") !== -1) {
            strOS = "Windows XP";
        } else if (strUserAgent.indexOf("windows 2000") !== -1) {
            strOS = "Windows 2000";
        } else if (strUserAgent.indexOf("windows nt 5.0") !== -1) {
            strOS = "Windows 2000";
        } else if (strUserAgent.indexOf("windows nt 4.0") !== -1) {
            strOS = "Windows NT 4.0";
        } else if (strUserAgent.indexOf("windows nt") !== -1) {
            strOS = "Windows NT 4.0";
        } else if (strUserAgent.indexOf("winnt4.0") !== -1) {
            strOS = "Windows NT 4.0";
        } else if (strUserAgent.indexOf("winnt") !== -1) {
            strOS = "Windows NT 4.0";
        } else if (strUserAgent.indexOf("windows me") !== -1) {
            strOS = "Windows ME";
        } else if (strUserAgent.indexOf("win 9x 4.90") !== -1) {
            strOS = "Windows ME";
        } else if (strUserAgent.indexOf("windows 98") !== -1) {
            strOS = "Windows 98";
        } else if (strUserAgent.indexOf("win98") !== -1) {
            strOS = "Windows 98";
        } else if (strUserAgent.indexOf("windows 95") !== -1) {
            strOS = "Windows 95";
        } else if (strUserAgent.indexOf("windows_95") !== -1) {
            strOS = "Windows 95";
        } else if (strUserAgent.indexOf("win95") !== -1) {
            strOS = "Windows 95";
        } else if (strUserAgent.indexOf("ce") !== -1) {
            strOS = "Windows CE";
        } else if (strUserAgent.indexOf("win16") !== -1) {
            strOS = "Windows 3.11";
        } else if (strUserAgent.indexOf("iemobile") !== -1) {
            strOS = "Windows Mobile";
        } else if (strUserAgent.indexOf("wm5 pie") !== -1) {
            strOS = "Windows Mobile";
        } else if (strUserAgent.indexOf("windows") !== -1) {
            strOS = "Windows (Unknown Version)";
        } else if (strUserAgent.indexOf("openbsd") !== -1) {
            strOS = "Open BSD";
        } else if (strUserAgent.indexOf("sunos") !== -1) {
            strOS = "Sun OS";
        } else if (strUserAgent.indexOf("ubuntu") !== -1) {
            strOS = "Ubuntu";
        } else if (strUserAgent.indexOf("ipad") !== -1) {
            strOS = "iOS (iPad)";
        } else if (strUserAgent.indexOf("ipod") !== -1) {
            strOS = "iOS (iTouch)";
        } else if (strUserAgent.indexOf("iphone") !== -1) {
            strOS = "iOS (iPhone)";
        } else if (strUserAgent.indexOf("mac os x beta") !== -1) {
            strOS = "Mac OSX Beta (Kodiak)";
        } else if (strUserAgent.indexOf("mac os x 10.0") !== -1) {
            strOS = "Mac OSX Cheetah";
        } else if (strUserAgent.indexOf("mac os x 10.1") !== -1) {
            strOS = "Mac OSX Puma";
        } else if (strUserAgent.indexOf("mac os x 10.2") !== -1) {
            strOS = "Mac OSX Jaguar";
        } else if (strUserAgent.indexOf("mac os x 10.3") !== -1) {
            strOS = "Mac OSX Panther";
        } else if (strUserAgent.indexOf("mac os x 10.4") !== -1) {
            strOS = "Mac OSX Tiger";
        } else if (strUserAgent.indexOf("mac os x 10.5") !== -1) {
            strOS = "Mac OSX Leopard";
        } else if (strUserAgent.indexOf("mac os x 10.6") !== -1) {
            strOS = "Mac OSX Snow Leopard";
        } else if (strUserAgent.indexOf("mac os x 10.7") !== -1) {
            strOS = "Mac OSX Lion";
        } else if (strUserAgent.indexOf("mac os x") !== -1) {
            strOS = "Mac OSX (Version Unknown)";
        } else if (strUserAgent.indexOf("mac_68000") !== -1) {
            strOS = "Mac OS Classic (68000)";
        } else if (strUserAgent.indexOf("68K") !== -1) {
            strOS = "Mac OS Classic (68000)";
        } else if (strUserAgent.indexOf("mac_powerpc") !== -1) {
            strOS = "Mac OS Classic (PowerPC)";
        } else if (strUserAgent.indexOf("ppc mac") !== -1) {
            strOS = "Mac OS Classic (PowerPC)";
        } else if (strUserAgent.indexOf("macintosh") !== -1) {
            strOS = "Mac OS Classic";
        } else if (strUserAgent.indexOf("googletv") !== -1) {
            strOS = "Android (GoogleTV)";
        } else if (strUserAgent.indexOf("xoom") !== -1) {
            strOS = "Android (Xoom)";
        } else if (strUserAgent.indexOf("htc_flyer") !== -1) {
            strOS = "Android (HTC Flyer)";
        } else if (strUserAgent.indexOf("android") !== -1) {
            strOS = "Android";
        } else if (strUserAgent.indexOf("symbian") !== -1) {
            strOS = "Symbian";
        } else if (strUserAgent.indexOf("series60") !== -1) {
            strOS = "Symbian (Series 60)";
        } else if (strUserAgent.indexOf("series70") !== -1) {
            strOS = "Symbian (Series 70)";
        } else if (strUserAgent.indexOf("series80") !== -1) {
            strOS = "Symbian (Series 80)";
        } else if (strUserAgent.indexOf("series90") !== -1) {
            strOS = "Symbian (Series 90)";
        } else if (strUserAgent.indexOf("x11") !== -1) {
            strOS = "UNIX";
        } else if (strUserAgent.indexOf("nix") !== -1) {
            strOS = "UNIX";
        } else if (strUserAgent.indexOf("linux") !== -1) {
            strOS = "Linux";
        } else if (strUserAgent.indexOf("qnx") !== -1) {
            strOS = "QNX";
        } else if (strUserAgent.indexOf("os/2") !== -1) {
            strOS = "IBM OS/2";
        } else if (strUserAgent.indexOf("beos") !== -1) {
            strOS = "BeOS";
        } else if (strUserAgent.indexOf("blackberry95") !== -1) {
            strOS = "Blackberry (Storm 1/2)";
        } else if (strUserAgent.indexOf("blackberry97") !== -1) {
            strOS = "Blackberry (Bold)";
        } else if (strUserAgent.indexOf("blackberry96") !== -1) {
            strOS = "Blackberry (Tour)";
        } else if (strUserAgent.indexOf("blackberry89") !== -1) {
            strOS = "Blackberry (Curve 2)";
        } else if (strUserAgent.indexOf("blackberry98") !== -1) {
            strOS = "Blackberry (Torch)";
        } else if (strUserAgent.indexOf("playbook") !== -1) {
            strOS = "Blackberry (Playbook)";
        } else if (strUserAgent.indexOf("wnd.rim") !== -1) {
            strOS = "Blackberry (IE/FF Emulator)";
        } else if (strUserAgent.indexOf("blackberry") !== -1) {
            strOS = "Blackberry";
        } else if (strUserAgent.indexOf("palm") !== -1) {
            strOS = "Palm OS";
        } else if (strUserAgent.indexOf("webos") !== -1) {
            strOS = "WebOS";
        } else if (strUserAgent.indexOf("hpwos") !== -1) {
            strOS = "WebOS (HP)";
        } else if (strUserAgent.indexOf("blazer") !== -1) {
            strOS = "Palm OS (Blazer)";
        } else if (strUserAgent.indexOf("xiino") !== -1) {
            strOS = "Palm OS (Xiino)";
        } else if (strUserAgent.indexOf("kindle") !== -1) {
            strOS = "Kindle";
        } else if (strUserAgent.indexOf("wii") !== -1) {
            strOS = "Nintendo (Wii)";
        } else if (strUserAgent.indexOf("nintendo ds") !== -1) {
            strOS = "Nintendo (DS)";
        } else if (strUserAgent.indexOf("playstation 3") !== -1) {
            strOS = "Sony (Playstation Console)";
        } else if (strUserAgent.indexOf("playstation portable") !== -1) {
            strOS = "Sony (Playstation Portable)";
        } else if (strUserAgent.indexOf("webtv") !== -1) {
            strOS = "MSN TV (WebTV)";
        } else if (strUserAgent.indexOf("inferno") !== -1) {
            strOS = "Inferno";
        } else {
            strOS = "Unknown";
        }
        if (strPlatform.indexOf("x64") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("wow64") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("win64") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("win32") !== -1) {
            strOSBits = "32 bits";
        } else if (strPlatform.indexOf("x64") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("x32") !== -1) {
            strOSBits = "32 bits";
        } else if (strPlatform.indexOf("x86") !== -1) {
            strOSBits = "32 bits*";
        } else if (strPlatform.indexOf("ppc") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("alpha") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("68k") !== -1) {
            strOSBits = "64 bits";
        } else if (strPlatform.indexOf("iphone") !== -1) {
            strOSBits = "32 bits";
        } else if (strPlatform.indexOf("android") !== -1) {
            strOSBits = "32 bits";
        } else {
            strOSBits = "Unknown";
        }
        strOut = strOS + strSep + strOSBits;
        return strOut;
    } catch (err) {
        return strOnError;
    }
}


function fingerprint_useragent() {
    "use strict";
    var strSep, strTmp, strUserAgent, strOut;

    strSep = "|";
    strTmp = null;
    strUserAgent = null;
    strOut = null;

    strUserAgent = navigator.userAgent.toLowerCase();
    strTmp = strUserAgent + strSep + navigator.platform;
    if (navigator.cpuClass) {
        strTmp += strSep + navigator.cpuClass;
    }
    if (navigator.browserLanguage) {
        strTmp += strSep + navigator.browserLanguage;
    } else {
        strTmp += strSep + navigator.language;
    }
    strOut = strTmp;
    return strOut;
}

function fingerprint_timezone() {
    "use strict";
    var strOnError, dtDate, numOffset, numGMTHours, numOut;

    strOnError = "Error";
    dtDate = null;
    numOffset = null;
    numGMTHours = null;
    numOut = null;

    try {
        dtDate = new Date();
        numOffset = dtDate.getTimezoneOffset();
        numGMTHours = (numOffset / 60) * (-1);
        numOut = numGMTHours;
        return numOut;
    } catch (err) {
        return strOnError;
    }
}

function murmurhash3_32_gc(key, seed) {
            var remainder, bytes, h1, h1b, c1, c2, k1, i;
            remainder = key.length & 3;
            bytes = key.length - remainder;
            h1 = seed;
            c1 = 0xcc9e2d51;
            c2 = 0x1b873593;
            i = 0;
            while (i < bytes) {
                k1 = ((key.charCodeAt(i) & 0xff)) | ((key.charCodeAt(++i) & 0xff) << 8) | ((key.charCodeAt(++i) & 0xff) << 16) | ((key.charCodeAt(++i) & 0xff) << 24);
                ++i;
                k1 = ((((k1 & 0xffff) * c1) + ((((k1 >>> 16) * c1) & 0xffff) << 16))) & 0xffffffff;
                k1 = (k1 << 15) | (k1 >>> 17);
                k1 = ((((k1 & 0xffff) * c2) + ((((k1 >>> 16) * c2) & 0xffff) << 16))) & 0xffffffff;
                h1 ^= k1;
                h1 = (h1 << 13) | (h1 >>> 19);
                h1b = ((((h1 & 0xffff) * 5) + ((((h1 >>> 16) * 5) & 0xffff) << 16))) & 0xffffffff;
                h1 = (((h1b & 0xffff) + 0x6b64) + ((((h1b >>> 16) + 0xe654) & 0xffff) << 16));
            }
            k1 = 0;
            switch (remainder) {
                case 3:
                    k1 ^= (key.charCodeAt(i + 2) & 0xff) << 16;
                case 2:
                    k1 ^= (key.charCodeAt(i + 1) & 0xff) << 8;
                case 1:
                    k1 ^= (key.charCodeAt(i) & 0xff);
                    k1 = (((k1 & 0xffff) * c1) + ((((k1 >>> 16) * c1) & 0xffff) << 16)) & 0xffffffff;
                    k1 = (k1 << 15) | (k1 >>> 17);
                    k1 = (((k1 & 0xffff) * c2) + ((((k1 >>> 16) * c2) & 0xffff) << 16)) & 0xffffffff;
                    h1 ^= k1;
            }
            h1 ^= key.length;
            h1 ^= h1 >>> 16;
            h1 = (((h1 & 0xffff) * 0x85ebca6b) + ((((h1 >>> 16) * 0x85ebca6b) & 0xffff) << 16)) & 0xffffffff;
            h1 ^= h1 >>> 13;
            h1 = ((((h1 & 0xffff) * 0xc2b2ae35) + ((((h1 >>> 16) * 0xc2b2ae35) & 0xffff) << 16))) & 0xffffffff;
            h1 ^= h1 >>> 16;
            return h1 >>> 0;
        }


var browser_name = fingerprint_browser();
var res = fingerprint_display();
var os_version = fingerprint_os();
var time_zone = fingerprint_timezone();
var useragent = fingerprint_useragent();
// detect isMobile or Desktop
var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
var platform = "";
if (isMobile){
        platform = "Mobile";}
else {
        platform = "Desktop";}
//

var key = browser_name+"#"+res+"#"+os_version+"#"+time_zone+"#"+useragent;

var uid = murmurhash3_32_gc(key, 31);

// IP address
var info = fetch('https://api.ipify.org/?format=json').then(response => response.json());
info = Promise.resolve(info);
info.then(function(result)  {
		const ips = result["ip"];
		// sending to python server
		// uid is the hash (murmur3_32 hash)
		var xhr = new XMLHttpRequest();
		xhr.open("POST", "/json_data", true);
		xhr.setRequestHeader('Content-Type', 'application/json');
		xhr.send(JSON.stringify({
			ip: ips,
			uid: uid,  
			browser: browser_name,
			resolution: res,
			os: os_version,
			timezone: time_zone,
			user_agent: useragent,
			platform: platform,
			platformmodel: navigator.platform,
			language: navigator.language,
			vendor: navigator.vendor,
			vendorSub: navigator.vendorSub,
		})); 
	});

console.log(uid);
console.log(fingerprint_browser());
console.log(fingerprint_display());
console.log(fingerprint_os());
console.log(fingerprint_timezone());
console.log(fingerprint_useragent());
console.log(navigator.language);
console.log(navigator.vendor);
console.log(navigator.vendorSub);
