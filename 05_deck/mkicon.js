const React=require('react');
const {renderToStaticMarkup}=require('react-dom/server');
const md=require('react-icons/md');
const sharp=require('sharp');
const S=Number(process.argv[3]||256);
const inner=renderToStaticMarkup(React.createElement(md[process.argv[2]],{color:'#FFFFFF',size:S*0.62}));
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}"><rect width="${S}" height="${S}" rx="${S*0.18}" fill="#0B4EA2"/><g transform="translate(${S*0.19},${S*0.19})">${inner}</g></svg>`;
sharp(Buffer.from(svg)).png().toFile(process.argv[4]).then(()=>console.log('ok'));
